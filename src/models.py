import numpy as np
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
        self.num_layers = 1
        #Use VGG, NOTE: param.requires_grad are set to True by default
        self.vgg = models.vgg19(pretrained=True)
        num_ftrs = self.vgg.classifier[6].in_features
        #Changes the size of VGG output to the GRU size
        self.vgg.classifier[6] = nn.Linear(num_ftrs, args.ind_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers)

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
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, args, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = args.ind_size
        self.dropout_p = args.dropout
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.num_layers = 1
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        #self.attn = nn.Linear(self.hidden_size * 2, args.num_frames)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, input_lengths, encoder_outputs):

        drop_input = self.dropout(input)#.view(args.max_length, args.batch_size, -1))

        enc_perm = encoder_outputs.permute(1,2,0)
        drop_input_perm = drop_input.permute(1,0,2)
        dot_attn_weights = torch.bmm(drop_input_perm, enc_perm)
        normalized_dot_attn_weights = F.softmax(dot_attn_weights, dim=2)
        enc_perm_2 = encoder_outputs.permute(1,0,2)
        attn_applied = torch.bmm(normalized_dot_attn_weights, enc_perm_2).permute(1,0,2)

        output = torch.cat((drop_input, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)

        #output: (max_length, batch_size, ind_size)
        #hidden: (1, batch_size, ind_size)

        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        output =  output[:, perm_idx]

        packed = torch.nn.utils.rnn.pack_padded_sequence(output, input_lengths.int())
        packed, hidden = self.gru(packed, hidden)
        #undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)

        output = self.out(output)

        #output: (max_length, batch_size, ind_size)
        return output, hidden, perm_idx

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


class NumIndEOS(nn.Module):
    def __init__(self, args, device):
        super(NumIndEOS, self).__init__()
        self.device = device
        self.hidden_size = args.ind_size
        self.output_size = 1
        self.dropout_p = args.dropout
        self.max_length = args.max_length
        self.batch_size = args.batch_size

        self.attn = nn.Linear(self.hidden_size * 2, args.num_frames)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    #def forward(self, input, hidden, encoder_outputs):
    def forward(self, input, hidden, input_lengths, encoder_outputs):

        drop_input = self.dropout(input)#.view(args.max_length, args.batch_size, -1))

        enc_perm = encoder_outputs.permute(1,2,0)

        drop_input_perm = drop_input.permute(1,0, 2)
        dot_attn_weights = F.softmax(torch.bmm(drop_input_perm, enc_perm), dim=2)
        enc_perm_2 = encoder_outputs.permute(1,0,2)
        attn_applied = torch.bmm(dot_attn_weights, enc_perm_2).permute(1,0,2)

        output = torch.cat((drop_input, attn_applied), 2)
        #print('output', output.shape)
        output = self.attn_combine(output)

        output = torch.sigmoid(output)
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        output =  output[:, perm_idx]

        packed = torch.nn.utils.rnn.pack_padded_sequence(output, input_lengths.int())
        packed, hidden = self.gru(packed, hidden)
        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)
        #output, hidden = self.gru(output, hidden)

        #print('output', output.shape)
        #output = F.log_softmax(self.out(output[0]), dim=1)

        output = self.out(output)
        output = torch.sigmoid(output)

        #print('output', output.shape)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


class NumIndRegressor(nn.Module):
    def __init__(self, args, device):
        super(NumIndRegressor, self).__init__()
        self.device = device
        self.output_size = 1
        self.input_size = args.ind_size
        self.l1 = nn.Linear(self.input_size, int(self.input_size/2))
        self.l2 = nn.Linear(int(self.input_size/2), int(self.input_size/4))
        self.out = nn.Linear(int(self.input_size/4), self.output_size)

    # The input is supposed to be all the outputs of the encoder
    def forward(self, input):
        emb_sum = torch.sum(input, dim=0)
        l1_out = self.l1(emb_sum).squeeze(1)
        l1_out = F.relu(l1_out)
        l2_out = self.l2(l1_out)
        l2_out = F.relu(l2_out)
        output = self.out(l2_out)
        return output


def train_seq2seq_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    print(device)
    
    encoder_hidden = encoder.initHidden().to(device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    encoder_outputs = encoder_outputs.to(device)
    encoder_hidden = encoder_hidden.to(device)
    decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device).to(device)

    decoder_hidden = encoder_hidden[encoder.num_layers-1:encoder.num_layers]
    decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
    decoder_outputs, decoder_hidden, perm_idx = decoder(input=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden)
    
    target_tensor = target_tensor[:, perm_idx]

    tot_elements = torch.sum(target_number_tensor)

    arange = torch.arange(0, decoder_outputs.shape[0], step=1).expand(args.ind_size, args.batch_size, decoder_outputs.shape[0]).cuda()
    arange = arange.permute(2, 1, 0)

    lengths = target_number_tensor.expand(decoder_outputs.shape[0], args.ind_size, args.batch_size).long().cuda()
    lengths = lengths.permute(0, 2, 1)

    # print("arange")
    # print(arange.shape)
    # print("lengths")
    # print(lengths.shape)
    # print("decoder outputs")
    # print(decoder_outputs.shape)
    # print("target_tensor[:decoder_outputs.shape[0],:,:]")
    # print(target_tensor[:decoder_outputs.shape[0],:,:].shape)

    mask = arange < lengths 
    mask = mask.cuda().float()

    loss = criterion(decoder_outputs*mask, target_tensor[:decoder_outputs.shape[0],:,:]*mask)
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train_reg_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, regressor, optimizer, criterion, device):

    encoder_hidden = encoder.initHidden().to(device)
    optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    encoder_outputs = encoder_outputs.to(device)
    encoder_hidden = encoder_hidden.to(device)

    """
    if torch.cuda.is_available():
        encoder_hidden = encoder_hidden.cuda()
        encoder_outputs = encoder_outputs.cuda()
    """

    regressor_output = regressor(encoder_outputs)
    loss = criterion(regressor_output, target_number_tensor)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_eos_on_batch(args, input_tensor, target_tensor, target_number_tensor, eos_target, encoder, eos, optimizer, criterion, device):
    
    encoder_hidden = encoder.initHidden().to(device)
    optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    eos_input = torch.zeros(1, args.batch_size, args.ind_size, device=eos.device).to(device)
    
    encoder_outputs = encoder_outputs.to(device)
    encoder_hidden = encoder_hidden.to(device)
    
    #if torch.cuda.is_available():
    #    encoder_hidden = encoder_hidden.cuda()
    #    encoder_outputs = encoder_outputs.cuda()
    #    eos_input = eos_input.cuda()

    eos_hidden = encoder_hidden[encoder.num_layers-1:encoder.num_layers]
    eos_inputs = torch.cat((eos_input, target_tensor[:-1]))
    outputs, hidden = eos(input=eos_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=eos_hidden)
    
    loss = criterion(outputs.squeeze(2), eos_target[:outputs.shape[0],:])
    loss.backward()
    optimizer.step()

    return loss.item()


def eval_network_on_batch(mode, args, input_tensor, target_tensor, target_number_tensor=None, eos_target=None, encoder=None, decoder=None, regressor=None, eos=None, dec_criterion=None, reg_criterion=None, eos_criterion=None, device='cpu'):
    """Possible values for 'mode' arg: {"eval_seq2seq", "eval_reg", "eval_eos", "test"}"""
    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    if mode == "eval_reg":
        regressor_output = regressor(encoder_outputs)
        reg_loss = reg_criterion(regressor_output, target_number_tensor)
        return reg_loss.item()

    elif mode == "eval_seq2seq":
        decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device).to(device)
        decoder_hidden_0 = encoder_hidden[encoder.num_layers-1:encoder.num_layers]
        #if torch.cuda.is_available():
            #decoder_input = decoder_input.cuda()
        dec_loss = 0
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, decoder_hidden, perm_idx = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden) #input_lengths=torch.tensor([target_number_tensor[b]])
                l_loss += dec_criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                single_dec_input = decoder_output
            dec_loss += l_loss/float(l)

        dec_loss /= float(b) 
        return dec_loss.item()

    elif mode == "eval_eos":
        eos_input = torch.zeros(1, args.batch_size, args.ind_size, device=eos.device).to(device)
        eos_hidden = encoder_hidden[encoder.num_layers-1:encoder.num_layers]
        eos_inputs = torch.cat((eos_input, target_tensor[:-1]))
        eos_outputs, hidden = eos(input=eos_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=eos_hidden)
        eos_loss = eos_criterion(eos_outputs.squeeze(2), eos_target[:eos_outputs.shape[0],:])
        return eos_loss.item()
    
    elif mode == "test":
        if encoder == None or decoder==None or (eos==None and regressor==None):
            print("Trained model necessary for testing. Input encoder, decoder and either eos or regressor")
            exit()
        regressor_output = regressor(encoder_outputs)
        reg_loss = reg_criterion(regressor_output, target_number_tensor)
        dec_loss = 0
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b]
            for l in range(target_number_tensor[b].int()):
                decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.ones(1), encoder_outputs=encoder_outputs, hidden=decoder_hidden)
                dec_loss += criterion(decoder_output, target_tensor[l, b])
            dec_loss = dec_loss / float(l)
        dec_loss /= float(b) 
        return dec_loss.item(), reg_loss.item()


def get_test_output(checkpoint_path, input_tensor, num_datapoints, ind_size, use_eos=True, device='cpu'):

    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    if regressor == None and eos==None:
        print("Either regressor or eos must be specified")
        exit()

    if use_eos:

        eos = checkpoint['eos']

        encoder_hidden = encoder.initHidden().to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.zeros(1, 1, ind_size, device=decoder.device).to(device)
        eos_input = torch.zeros(1, 1, ind_size, device=eos.device).to(device)

        print("input tensor")
        print(input_tensor.shape)

        for d in num_datapoints:
            print("EHI")

    
def train_iters_seq2seq(args, encoder, decoder, train_generator, val_generator, exp_name, device):
    print(device)

    loss_plot_file_path = '../data/loss_plots/loss_seq2seq{}.png'.format(exp_name)

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

    # Freeze vgg layers that we don't want to train
    v = 1
    for param in encoder.vgg.parameters():
        if v <= args.vgg_layers_to_freeze*2: # Assuming each layer has two params
            param.requires_grad = False
        v += 1

    epoch_train_losses = []
    epoch_val_losses = []
    for epoch_num in range(args.max_epochs):
        batch_train_losses = []
        print("Epoch:", epoch_num+1)

        batch_val_losses = []
        
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            new_val_loss = eval_network_on_batch("eval_seq2seq", args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, dec_criterion=criterion, device=device)
            batch_val_losses.append(new_val_loss)

            if args.quick_run:
                break

        print(batch_train_losses, batch_val_losses)
        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_val_losses.append(new_epoch_val_loss)
        utils.plot_losses(epoch_train_losses, epoch_val_losses, loss_plot_file_path)
        save_dict = {'encoder':encoder, 'decoder':decoder}
        EarlyStop(new_epoch_val_loss, save_dict, filename='../checkpoints/chkpt{}.pt'.format(exp_name))
        if EarlyStop.early_stop:
            return EarlyStop.val_loss_min

        for iter_, training_triplet in enumerate(train_generator):
            
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            
            new_train_loss = train_seq2seq_on_batch(args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, criterion=criterion, device=device)
            print(iter_, new_train_loss)

            batch_train_losses.append(new_train_loss)
            if args.quick_run:
                break

    return new_epoch_val_loss


def train_iters_reg(args, encoder, decoder, regressor, train_generator, val_generator, exp_name, device):

    if args.optimizer == "SGD":
        optimizer = optim.SGD(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "RMS":
        optimizer = optim.RMSProp(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.MSELoss()
    EarlyStop = EarlyStopper(patience=args.patience, verbose=True)

    # Freeze encoder to save computing gradient
    for param in encoder.parameters():
        param.requires_grad = False
    
    loss_plot_file_path = '../data/loss_plots/loss_reg_{}.png'.format(exp_name)
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch_num in range(args.max_epochs):
        
        batch_train_losses = [] 
        print("Epoch:", epoch_num+1)
    
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            new_train_loss = train_reg_on_batch(args, input_tensor, target_tensor, target_number, encoder=encoder, regressor=regressor, optimizer=optimizer, criterion=criterion, device=device)

            print(iter_, new_train_loss)
            batch_train_losses.append(new_train_loss)
            if args.quick_run:
                break

        batch_val_losses =[] 
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            new_val_loss = eval_network_on_batch("eval_reg", args, input_tensor, target_tensor, target_number, encoder=encoder, regressor=regressor, reg_criterion=criterion, device=device)

            batch_val_losses.append(new_val_loss)
            if args.quick_run:
                break


        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_val_losses.append(new_epoch_val_loss)
        utils.plot_losses(epoch_train_losses, epoch_val_losses, loss_plot_file_path)
        save_dict =  {'encoder':encoder, 'decoder':decoder, 'regressor':regressor}
        EarlyStop(new_epoch_val_loss, save_dict, filename='../checkpoints/chkpt{}.pt'.format(exp_name))
        
        if EarlyStop.early_stop:
            return EarlyStop.val_loss_min

    return new_epoch_val_loss

def train_iters_eos(args, encoder, decoder, eos, train_generator, val_generator, exp_name, device):

    if args.optimizer == "SGD":
        optimizer = optim.SGD(eos.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(eos.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "RMS":
        optimizer = optim.RMSProp(eos.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.MSELoss()
    EarlyStop = EarlyStopper(patience=args.patience, verbose=True)

    # Freeze encoder to save computing gradient
    for param in encoder.parameters():
        param.requires_grad = False
        
    loss_plot_file_path = '../data/loss_plots/loss_eos_{}.png'.format(exp_name)
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch_num in range(args.max_epochs):
        batch_train_losses = []
        print("Epoch:", epoch_num+1)
    
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            eos_target = training_triplet[3].float().permute(1,0).to(device)
            new_train_loss = train_eos_on_batch(args, input_tensor, target_tensor, target_number, eos_target, encoder=encoder, eos=eos, optimizer=optimizer, criterion=criterion, device=device)

            print(iter_, new_train_loss)
            batch_train_losses.append(new_train_loss)

            if args.quick_run:
                break

        batch_val_losses =[] 
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            eos_target = training_triplet[3].float().permute(1,0).to(device)
            new_val_loss = eval_network_on_batch("eval_eos", args, input_tensor, target_tensor, target_number, eos_target, encoder=encoder, eos=eos, eos_criterion=criterion, device=device)
            batch_val_losses.append(new_val_loss)

            if args.quick_run:
                break

        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_val_losses.append(new_epoch_val_loss)
        utils.plot_losses(epoch_train_losses, epoch_val_losses, loss_plot_file_path)
        save_dict = {'encoder':encoder, 'decoder':decoder, 'eos': eos}
        EarlyStop(new_epoch_val_loss, save_dict, filename='../checkpoints/chkpt{}.pt'.format(exp_name))
        if EarlyStop.early_stop:
            return EarlyStop.val_loss_min

    return new_epoch_val_loss



if __name__ == "__main__":
    import options
    args = options.load_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device available:")
    print(device)
    eos = NumIndEOS(args, device)
    #eos = DecoderRNN(args, device)
    eos.cuda()
    il = torch.ones(size=(args.batch_size,))

    hidden = eos.initHidden().cuda()
    inp = torch.ones(size=(10,args.batch_size,300)).cuda()
    enc_out = torch.ones(size=(8, args.batch_size,300)).cuda()
    #outp = eos(inp, hidden, il, enc_out).cuda()
    outp, _ = eos(inp, hidden, enc_out)
    print(outp.shape)

    """
    enc = EncoderRNN(args, device)
    v = 0
    for param in enc.vgg.parameters():
        v += 1
        print(param.requires_grad)
        if v <= 34:
            param.requires_grad = False
    e = 0 
    g = 0
    for param in enc.parameters():
        e += 1
        if param.requires_grad:
            g +=1 
   
    print(e,v,g)

   """ 
