import random
import re
import string
import time
import unicodedata
import utils

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from early_stopper import EarlyStopper


class EncoderRNN(nn.Module):
    def __init__(self, args, device):
        super(EncoderRNN, self).__init__()
        self.num_frames = args.num_frames
        self.hidden_size = args.ind_size
        self.device = device
        self.batch_size = args.batch_size
        #Use VGG, NOTE: param.requires_grad are set to True by default
        self.vgg = models.vgg19(pretrained=True)
        num_ftrs = self.vgg.classifier[6].in_features
        #Changes the size of VGG output to the GRU size
        self.vgg.classifier[6] = nn.Linear(num_ftrs, args.ind_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):

        # if we pass entire sequences we have to reset the GRU hidden state. Otherwise it'll see the new batch as a continuation of a sequence
        hidden = self.initHidden()

        vgg_outputs = torch.zeros(self.num_frames, self.batch_size, self.hidden_size, device=self.device)

        for i, inp in enumerate(input):
            embedded = self.vgg(inp)#.view(1, self.batch_size, -1)
            vgg_outputs[i] = embedded 

        #outputs: (num_frames, batch_size, ind_size)
        #hidden: (1, batch_size, ind_size)
        outputs, hidden = self.gru(vgg_outputs, hidden)

        return outputs, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, args, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = args.ind_size
        self.dropout_p = args.dropout
        self.max_length = args.max_length
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        #self.attn = nn.Linear(self.hidden_size * 2, args.num_frames)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, input_lengths, encoder_outputs):

        drop_input = self.dropout(input)#.view(args.max_length, args.batch_size, -1))

        enc_perm = encoder_outputs.permute(1,2,0)
        #print('enc_perm', enc_perm.shape)
        drop_input_perm = drop_input.permute(1,0, 2)
        #print('drop_perm', drop_input_perm.shape)
        dot_attn_weights = torch.bmm(drop_input_perm, enc_perm)
        #print('dot_attn', dot_attn_weights.shape)
        enc_perm_2 = encoder_outputs.permute(1,0,2)
        #print('enc_perm_2', enc_perm_2.shape)
    #     attn_weights = F.softmax(
    #         self.attn(torch.cat((drop_input[0], hidden[0]), 1)), dim=1)
    #     #attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        attn_applied = torch.bmm(dot_attn_weights, enc_perm_2).permute(1,0,2)
        #print('attn_applied', attn_applied.shape)


        output = torch.cat((drop_input, attn_applied), 2)
        #print('output', output.shape)
        output = self.attn_combine(output)
        #print('output', output.shape)

        output = F.relu(output)
    #     output, hidden = self.gru(output, hidden)

        #output: (max_length, batch_size, ind_size)
        #hidden: (1, batch_size, ind_size)

        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        output =  output[:, perm_idx]

        packed = torch.nn.utils.rnn.pack_padded_sequence(output, input_lengths.int())
        packed, hidden = self.gru(packed, hidden)
        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)
        #output = torch.cuda.FloatTensor(10,2,300).fill_(0)

        #output: (max_length, batch_size, ind_size)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


class NumIndRegressor(nn.Module):
    def __init__(self, args, device):
        super(NumIndRegressor, self).__init__()
        self.device = device
        self.output_size = 1
        self.input_size = args.ind_size
        self.out = nn.Linear(self.input_size, self.output_size)

    # The input is supposed to be all the outputs of the encoder
    def forward(self, input):
        emb_sum = torch.sum(input, dim=0)
        output = self.out(emb_sum).squeeze(1)
        return output


class NumIndEOS(nn.Module):
    def __init__(self, args, device):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = args.ind_size
        self.output_size = 1
        self.dropout_p = args.dropout
        self.max_length = args.max_length

        self.attn = nn.Linear(self.hidden_size * 2, args.num_frames)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        drop_input = self.dropout(input.view(1, 1, -1))

        attn_weights = F.softmax(
            self.attn(torch.cat((drop_input[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((drop_input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        #output = F.log_softmax(self.out(output[0]), dim=1)

        output = self.out(output[0])

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)



def train_seq2seq_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device)
    if torch.cuda.is_available():
        encoder_hidden = encoder_hidden.cuda()
        encoder_outputs = encoder_outputs.cuda()
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden
    decoder_inputs = torch.cat((decoder_input, target_tensor))
    decoder_outputs, decoder_hidden = decoder(input=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden)
    
    loss = criterion(decoder_outputs, target_tensor[:decoder_outputs.shape[0],:,:])
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train_reg_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, decoder, regressor, encoder_optimizer, decoder_optimizer, regressor_optimizer, dec_criterion, reg_criterion):

    encoder_hidden = encoder.initHidden()
    optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    if torch.cuda.is_available():
        encoder_hidden = encoder_hidden.cuda()
        encoder_outputs = encoder_outputs.cuda()

    regressor_output = regressor(encoder_outputs)
    loss += reg_criterion(regressor_output, target_number_tensor)

    loss.backward()
    optimizer.step()

    return loss.item()


def eval_network_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, decoder, regressor, criterion, reg_criterion, mode):
    """Possible values for 'mode' arg: {"eval_seq2seq", "eval_reg", "test"}"""
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    regressor_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device)
    
    if torch.cuda.is_available():
        decoder_input = decoder_input.cuda()
        encoder_hidden = encoder_hidden.cuda()
        encoder_outputs = encoder_outputs.cuda()

    decoder_hidden = encoder_hidden

    if mode == "eval_reg":
        regressor_output = regressor(encoder_outputs)
        reg_loss = reg_criterion(regressor_output, target_number_tensor)
        return reg_loss

    elif mode == "eval_seq2seq":
        dec_loss = 0
        for b in args.batch_size:
            single_dec_input = decoder_input[:, b]
            for l in range(target_number_tensor):
                decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.ones(1), encoder_outputs=encoder_outputs, hidden=decoder_hidden)
                dec_loss += criterion(decoder_output, target_tensor[l, b])
            dec_loss = dec_loss / float(l)
        dec_loss /= float(b) 
        return dec_loss    

    elif mode == "test":
        regressor_output = regressor(encoder_outputs)
        reg_loss = reg_criterion(regressor_output, target_number_tensor)
        dec_loss = 0
        for b in args.batch_size:
            single_dec_input = decoder_input[:, b]
            for l in range(target_number_tensor):
                decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.ones(1), encoder_outputs=encoder_outputs, hidden=decoder_hidden)
                dec_loss += criterion(decoder_output, target_tensor[l, b])
            dec_loss = dec_loss / float(l)
        dec_loss /= float(b) 
        return dec_loss, reg_loss

    
def train_iters_seq2seq(args, encoder, decoder, regressor, train_generator, val_generator, print_every=1000, plot_every=1000, exp_name=""):

    start = time.time()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print(args.optimizer)
    if args.optimizer == "SGD":
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "RMS":
        encoder_optimizer = optim.RMSProp(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        decoder_optimizer = optim.RMSProp(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.MSELoss()
    EarlyStop = EarlyStopper(patience=args.patience, verbose=True)

    for epoch_num in range(args.max_epochs):
        print("Epoch:", epoch_num+1)
    
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1)
            target_tensor = training_triplet[1].float().transpose(0,1)
            target_number = training_triplet[2].float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
                target_number = target_number.cuda()
            loss = train_seq2seq_on_batch(args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, criterion=criterion)

            print_loss_total += loss
            plot_loss_total += loss

            if iter_ % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                #print('%s (%d %d%%) %.4f' % (utils.timeSince(start, iter_+1 / n_iters),
                                             #iter_+1, iter_+1 / n_iters * 100, print_loss_avg))

            if iter_ % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                utils.showPlot(plot_losses, '../data/loss_plots/loss{}.png'.format(exp_name))

        total_val_loss = 0
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1)
            target_tensor = training_triplet[1].float().transpose(0,1)
            target_number = training_triplet[2].float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
                target_number = target_number.cuda()
            new_val_loss = eval_network_on_batch(args, input_tensor, target_tensor, target_number, encoder, decoder, encoder_optimizer, decoder_optimizer, dec_criterion, mode="eval_seq2seq")

            total_val_loss += new_val_loss[0]
        save_dict = {
            'encoder':encoder, 
            'decoder':decoder, 
        }
        EarlyStop(total_val_loss, save_dict, filename='../checkpoints/chkpt{}.pt'.format(exp_name))
        if EarlyStop.early_stop:
            return EarlyStop.val_loss_min



def train_iters_reg(args, regressor, train_generator, val_generator, print_every=1000, plot_every=1000, exp_name=""):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    if args.optimizer == "SGD":
        optimizer = optim.SGD(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "RMS":
        optimizer = optim.RMSProp(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.MSELoss()
    EarlyStop = EarlyStopper(patience=args.patience, verbose=True)

    for epoch_num in range(args.max_epochs):
        print("Epoch:", epoch_num+1)
    
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1)
            target_tensor = training_triplet[1].float().transpose(0,1)
            target_number = training_triplet[2].float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
                target_number = target_number.cuda()
            loss = train_reg_on_batch(args, input_tensor, target_tensor, target_number, regressor=regressor, optimizer=optimizer, criterion=criterion)

            print_loss_total += loss[0]
            plot_loss_total += loss[0]

            if iter_ % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                #print('%s (%d %d%%) %.4f' % (utils.timeSince(start, iter_+1 / n_iters),
                                             #iter_+1, iter_+1 / n_iters * 100, print_loss_avg))

            if iter_ % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                utils.showPlot(plot_losses, '../data/loss_plots/loss{}.png'.format(exp_name))

        total_val_loss = 0
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1)
            target_tensor = training_triplet[1].float().transpose(0,1)
            target_number = training_triplet[2].float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
                target_number = target_number.cuda()
            new_val_loss = eval_network_on_batch(args, input_tensor, target_tensor, target_number, regressor=regressor, criterion=reg_criterion, mode="eval_reg")

            total_val_loss += new_val_loss[0]
        EarlyStop(total_val_loss, {
            'regressor':regressor},
            filename='../checkpoints/chkpt{}.pt'.format(exp_name))
        if EarlyStop.early_stop:
            return EarlyStop.val_loss_min


