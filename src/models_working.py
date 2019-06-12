import math
import numpy as np
import random
from random import shuffle
import re
import string
import time
import unicodedata
import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from early_stopper import EarlyStopper
from attention import Attention
import pretrainedmodels

##############################
#
# ENCODER
#
##############################

class EncoderRNN(nn.Module):
    
    def __init__(self, args, device):
        super(EncoderRNN, self).__init__()
        self.num_frames = args.num_frames
        self.output_cnn_size = args.output_cnn_size
        #self.output_cnn_size = 4096
        #self.hidden_size = args.ind_size
        self.hidden_size = args.enc_size
        self.device = device
        self.batch_size = args.batch_size
        self.num_layers = args.enc_layers
        self.rnn_type = args.enc_rnn
        self.cnn_type = args.enc_cnn

        #self.cnn = models.vgg19(pretrained=True)
        #num_ftrs = self.cnn.classifier[6].in_features
        #self.cnn.classifier[6] = nn.Linear(num_ftrs, self.output_cnn_size)
        
        self.vgg = models.vgg19(pretrained=True)
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_ftrs, self.output_cnn_size)

        
        #if args.enc_cnn in ["vgg", "vgg_old"]:
        #    self.cnn = models.vgg19(pretrained=True)
        #    num_ftrs = self.cnn.classifier[6].in_features
        #    self.cnn.classifier[6] = nn.Linear(num_ftrs, self.output_cnn_size)
        if args.enc_cnn == "vgg_old":
            self.cnn = models.vgg19(pretrained=True)
            num_ftrs = self.cnn.classifier[6].in_features
            self.cnn.classifier[6] = nn.Linear(num_ftrs, self.output_cnn_size)
        elif args.enc_cnn == "vgg":
            self.cnn = models.vgg19(pretrained=True)
        elif args.enc_cnn == "nasnet":
            model_name = 'nasnetalarge'
            self.cnn = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        else:
            print(args.enc_cnn)


        #Encoder Recurrent layer
        self.gru = nn.GRU(self.output_cnn_size, self.hidden_size, num_layers=self.num_layers)

        # Resize outputs to ind_size
        self.resize = nn.Linear(self.hidden_size, args.ind_size)
        self.cnn_resize = nn.Linear(4032, self.output_cnn_size)


    def cnn_vector(self, input_):
        if self.cnn_type == "vgg_old":
            x = self.cnn(input_)
            #x = self.vgg(input_)
        elif self.cnn_type == "vgg":
            x = self.cnn.features(input_)
            x = self.cnn.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.cnn.classifier[0](x)
            #print('vgg output size', x.size())
            #x = self.cnn_resize(x)
        elif self.cnn_type == "nasnet":
            x = self.cnn.features(input_)
            x = self.cnn.relu(x)
            #x = x.mean(2).mean(2)
            x = self.cnn.avg_pool(x)
            x = x.view(x.size(0),-1)
            #print('nasnet output size', x.size())
            #reshape = nn.Linear(x.shape[0], self.output_cnn_size).to(self.device)
            #x = self.cnn_resize(x)

        #return self.resize(x)
        return x


    def forward(self, input, hidden):

        # pass the input throught the vgg layers 
        #print(self.output_cnn_size, self.hidden_size, self.num_layers)
        cnn_outputs = torch.zeros(self.num_frames, input.shape[1], self.output_cnn_size, device=self.device)
        for i, inp in enumerate(input):
            #embedded = self.vgg(inp)#.view(1, self.batch_size, -1)
            embedded = self.cnn_vector(inp)
            #embedded = self.cnn(inp)
            cnn_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        #print('cnn_outputs', cnn_outputs.size())
        #print('hidden', hidden.size())
        outputs, hidden = self.gru(cnn_outputs, hidden)
        
        outputs = self.resize(outputs)

        #print(" \n NETWORK INPUTS (first element in batch) \n")
        #print(input[0,0,:,:,0])
        #print(" \n NETWORK INPUTS (second element in batch) \n")
        #print(input[0,1,:,:,0])
        #print(" \n VGG OUTPUTS (first element in batch) \n")
        #print(cnn_outputs[:, 0])
        #print(" \n VGG OUTPUTS (second element in batch) \n")
        #print(cnn_outputs[:, 1])
        #print(" \n OUTPUTS (first element in batch \n")
        #print(outputs[:, 0])
        #print(" \n OUTPUTS (second element in batch \n")
        #print(outputs[:, 1])

        return outputs, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)




##############################
#
# DECODER
#
##############################

class DecoderRNN(nn.Module):
    
    def __init__(self, args, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = args.dec_size
        self.max_length = args.max_length
        self.num_layers = args.dec_layers
        self.dropout_p = args.dropout
        self.batch_size = args.batch_size
        self.device = device

        self.gru = nn.GRU(args.ind_size, self.hidden_size, num_layers=self.num_layers)
        #self.attention = Attention(self.hidden_size)
        self.attention = Attention(args.ind_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.out1 = nn.Linear(self.hidden_size, args.ind_size)
        #self.out2 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.out3 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, input_lengths, encoder_outputs):
        # apply dropout
        #print('input', input)
        drop_input = self.dropout(input)
        
        #pack the sequence to avoid useless computations
        packed = torch.nn.utils.rnn.pack_padded_sequence(drop_input, input_lengths.int(), enforce_sorted=False)
        packed, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)

        # permute dimensions to have batch_size in the first
        #enc_perm shape: (batch_size, num_frames, ind_size)
        enc_perm = encoder_outputs.permute(1,0,2)
        #output_perm shape: (batch_size, max_length, ind_size)
        output_perm = output.permute(1,0,2)

        # apply attention
        #output_perm, attn = self.attention(output_perm, enc_perm)
        output = self.out1(output_perm)
        #print('output b4 attn', output)
        output, attn = self.attention(output, enc_perm)
        # permute back to (max_length, batch_size, ind_size)
        #output = output_perm.permute(1,0,2)

        #output = self.out2(output)
        #output = self.out3(output)
        # output = self.out4(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)




##############################
#
# TRAIN FUNCTIONS
#
##############################


def train_on_batch(args, input_tensor, target_tensor, target_number_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):

    #encoder.train()
    #decoder.train()

    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    encoder_outputs = encoder_outputs.to(device)
    encoder_hidden = encoder_hidden.to(device)

    decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device).to(device)


    if use_teacher_forcing:

        #if args.enc_layers == args.dec_layers:
        if args.enc_dec_hidden_init:
            decoder_hidden = encoder_hidden#.expand(decoder.num_layers, args.batch_size, args.ind_size)
        else:
            decoder_hidden = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size).to(device)
            #decoder_hidden = torch.zeros(args.dec_layers, args.batch_size, args.ind_size)
        decoder_inputs = torch.cat((decoder_input, target_tensor[:-1]))
        decoder_outputs, decoder_hidden = decoder(input=decoder_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=decoder_hidden.contiguous())

        #norms = [torch.norm(decoder_outputs[i][j]).item() for i in range(decoder.batch_size) for j in range(decoder_outputs.shape[1])]
        #for i in range(decoder.batch_size):
        #    print('batch', i)
        #    seq_len = int(target_number_tensor[i].item())
        #    print('length', seq_len)
        #    for j in range(seq_len):
        #        print('output', torch.norm(decoder_outputs[i][j]).item())
        norms = [torch.norm(decoder_outputs[i][j]).item() for i in range(decoder.batch_size) for j in range(int(target_number_tensor[i].item()))]
        norms1 = [n for n in norms if n>0.1]

        # Note: target_tensor is passed to the decoder with shape (length, batch_size, ind_size)
        # but it then needs to be permuted to be compared in the loss. 
        # Output of decoder size: (batch_size, length, ind_size)
        target_tensor_perm = target_tensor.permute(1,0,2)
        
        #arange = torch.arange(0, decoder_outputs.shape[1], step=1).expand(args.batch_size, decoder_outputs.shape[1]).cuda()
        arange = torch.arange(0, decoder_outputs.shape[1], step=1).expand(args.batch_size, -1).cuda()
        #print('arange', arange.shape)
        #print('tnumt', target_number_tensor.shape)
        lengths = target_number_tensor.expand(decoder_outputs.shape[1], args.batch_size).long().cuda()
        #print('lengths', lengths.shape)
        lengths = lengths.permute(1,0)
        #print('lengths', lengths.shape)
        mask = arange < lengths 
        mask = mask.cuda().float().unsqueeze(2)
        #print(arange)
        #print(lengths)
        #print(mask)

        #print('mask', mask.shape)
        #print('dec out', decoder_outputs.shape)
        #print(decoder_outputs*mask)
        #print(target_tensor_perm[:,:decoder_outputs.shape[1],:]*mask)
        loss = criterion(decoder_outputs*mask, target_tensor_perm[:,:decoder_outputs.shape[1],:]*mask)
        #loss = criterion(decoder_outputs, target_tensor_perm[:,:decoder_outputs.shape[1],:])
        mse_rescale = (args.ind_size*args.batch_size*decoder_outputs.shape[1])/torch.sum(target_number_tensor)
        loss = loss*mse_rescale
        
        inv_byte_mask = mask.byte()^1
        inv_mask = inv_byte_mask.float()
        norm_tensor = decoder_outputs*mask + inv_mask
        #print(norm_tensor)
        output_norms = torch.norm(norm_tensor, dim=-1)
        output_norms = torch.norm(decoder_outputs, dim=-1)
        mask = mask.squeeze(2)
        inv_mask = inv_mask.squeeze(2)
        print(output_norms)
        output_norms = output_norms*mask + inv_mask
        #print(norm_tensor.shape)
        #print(output_norms.shape)
        #print(norms)
        mean_norm = output_norms.mean()
        #norm_loss = torch.abs(mean_norm-1)
        norm_loss = F.relu(1-mean_norm)*mse_rescale
        print(mean_norm.item(), 'gives loss', norm_loss.item())
        total_loss = loss +  norm_loss*args.lmbda
        #print('\tcomputing loss between {} and {}, result: {}'.format(decoder_outputs.squeeze()[0,0].item(), target_tensor.squeeze()[0,0].item(), loss))
        #print(args.batch_size*decoder_outputs.shape[1]/torch.sum(target_number_tensor))

    else:

        loss = 0 
        decoder_hidden_0 = encoder_hidden#.expand(decoder.num_layers,args.batch_size, args.ind_size)
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) #input_lengths=torch.tensor([target_number_tensor[b]])
                l_loss += criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                single_dec_input = decoder_output
            loss += l_loss/float(l)
        loss /= float(b)

    total_loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), norm_loss.item(), norms1


def train(args, encoder, decoder, train_generator, val_generator, exp_name, device, encoder_optimizer=None, decoder_optimizer=None):
    # where to save the image of the loss funcitons
    loss_plot_file_path = '../data/loss_plots/loss_{}.png'.format(exp_name)
    checkpoint_path = '../checkpoints/chkpt{}.pt'.format(exp_name)

    v = 1
    for param in encoder.cnn.parameters():
        #if v <= args.cnn_layers_to_freeze*2: # Assuming each layer has two params
            #param.requires_grad = False
        param.requires_grad = False
        v += 1

    if encoder_optimizer == None:
        encoder_params = filter(lambda enc: enc.requires_grad, encoder.parameters())
        if args.optimizer == "SGD":
            encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "RMS":
            encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if decoder_optimizer == None:
        if args.optimizer == "SGD":
            decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "RMS":
            decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    criterion = nn.MSELoss()
    EarlyStop = EarlyStopper(patience=args.patience, verbose=True)

    # Freeze vgg layers that we don't want to train
    #for param in encoder.vgg.parameters():
    #    if v <= args.cnn_layers_to_freeze*2: # Assuming each layer has two params
    #        param.requires_grad = False
    #    v += 1
    #v = 1


    epoch_train_losses = []
    epoch_train_norm_losses = []
    epoch_val_losses = []
    epoch_val_norm_losses = []
    for epoch_num in range(args.max_epochs):
        total_norms = []
        total_val_norms = []
        encoder.train()
        decoder.train()
        batch_train_losses = []
        batch_train_norm_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_triplet in enumerate(train_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            video_id = training_triplet[4].float().to(device)
            #print('input tensor on train', input_tensor.squeeze()[0,0,0,0].item())
            new_train_loss, new_train_norm_loss, norms1 = train_on_batch(args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, criterion=criterion, device=device)
            #print(video_id)
            #print(target_tensor.squeeze()[0])
            print(iter_, new_train_loss, new_train_norm_loss)
            #print(input_tensor[0,0,0,0,0])
            total_norms += norms1
            
            batch_train_losses.append(new_train_loss)
            batch_train_norm_losses.append(new_train_norm_loss)
            if args.quick_run:
                break
        encoder.eval()
        decoder.eval()
        batch_val_losses = []
        batch_val_norm_losses = []
        for iter_, training_triplet in enumerate(val_generator):
            input_tensor = training_triplet[0].float().transpose(0,1).to(device)
            target_tensor = training_triplet[1].float().transpose(0,1).to(device)
            target_number = training_triplet[2].float().to(device)
            #print('input tensor on eval', input_tensor.squeeze()[0,0,0,0].item())
            new_val_loss, new_val_norm_loss, val_norms = eval_on_batch("eval_seq2seq", args, input_tensor, target_tensor, target_number, encoder=encoder, decoder=decoder, dec_criterion=criterion, device=device)
            batch_val_losses.append(new_val_loss)
            batch_val_norm_losses.append(new_val_norm_loss)
            print('val', iter_, new_val_loss, new_val_norm_loss)

            total_val_norms += val_norms
            if args.quick_run:
                break

        new_epoch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        new_epoch_train_norm_loss = sum(batch_train_norm_losses)/len(batch_train_norm_losses)
        try:
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
            new_epoch_val_loss = sum(batch_val_losses)/len(batch_val_losses)
       
        new_epoch_val_norm_loss = sum(batch_val_norm_losses)/len(batch_val_norm_losses)
        
        epoch_train_losses.append(new_epoch_train_loss)
        epoch_train_norm_losses.append(new_epoch_train_norm_loss)

        epoch_val_losses.append(new_epoch_val_loss)
        epoch_val_norm_losses.append(new_epoch_val_norm_loss)
        #utils.plot_losses(epoch_train_losses, epoch_val_losses, loss_plot_file_path)
        #utils.plot_losses(epoch_train_losses, epoch_val_losses, 'most_recent_lossplot.png')
    
        print('plotting')
        axes = plt.axes()
        axes.set_ylim([0,min(6.5, max(epoch_val_losses))])
        plt.plot(epoch_train_losses, label='Train Loss')
        plt.plot(epoch_train_norm_losses, label='Train Norm Loss')
        plt.plot(epoch_val_losses, label='Validation Loss')
        plt.plot(epoch_val_norm_losses, label='Validation Norm Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend(loc='upper right')
        plt.xticks(np.arange(0, len(epoch_train_losses), math.ceil(len(epoch_train_losses)/8)))#, int(.125*len(train_losses))))
        plt.savefig(loss_plot_file_path)
        plt.savefig('../data/loss_plots/most_recent_lossplot.png')
        #plt.show()
        plt.close()
        plt.clf()

        print(epoch_train_losses)
        print(epoch_val_losses)
        print(epoch_train_norm_losses)
        print(epoch_val_norm_losses)
        save_dict = {'encoder':encoder, 'decoder':decoder, 'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer}
        save = (new_epoch_val_loss < 2.1) and args.chkpt
        EarlyStop(new_epoch_val_loss, save_dict, filename=checkpoint_path, save=save)
        #checkpoint = torch.load(checkpoint_path)
        #reloaded_encoder = checkpoint['encoder']
        #reloaded_decoder = checkpoint['decoder']
        #reload_val_losses = []
        #for iter_, training_triplet in enumerate(val_generator):
        #    input_tensor = training_triplet[0].float().transpose(0,1).to(device)
        #    target_tensor = training_triplet[1].float().transpose(0,1).to(device)
        #    target_number = training_triplet[2].float().to(device)
        #    new_reload_val_loss = eval_on_batch("eval_seq2seq", args, input_tensor, target_tensor, target_number, encoder=reloaded_encoder, decoder=reloaded_decoder, dec_criterion=criterion, device=device)
        #    reload_val_losses.append(new_reload_val_loss)

        #    if args.quick_run:
        #        break

        #reload_val_loss = sum(reload_val_losses)/len(reload_val_losses)
        print('val_loss', new_epoch_val_loss)
        #print('reloaded_val_loss', reload_val_loss)
        if EarlyStop.early_stop:
            break 
    #total_norms = total_norms.detach().numpy()
    #total_val_norms = total_val_norms.detach().numpy()
    print('train norms', len(total_norms))
    print('val norms', len(total_val_norms))
    plt.xlim(min(total_norms)-.1, max(total_norms)+.1)
    bins = np.arange(0,2,.01)
    plt.hist(total_norms, bins=bins)
    plt.xlabel('l2')
    plt.ylabel('number of embeddings')
    plt.title("{}: train norms".format(args.exp_name))
    plt.savefig('../data/norm_plots/{}_train_norms.png'.format(exp_name))
    plt.clf()

    plt.xlim(min(total_val_norms)-.1, max(total_val_norms)+.1)
    bins = np.arange(0,2,.01)
    plt.hist(total_val_norms, bins=bins)
    plt.xlabel('l2')
    plt.ylabel('number of embeddings')
    plt.title('{}: val norms'.format(args.exp_name))
    plt.savefig('../data/norm_plots/{}_val_norms.png'.format(exp_name))
    plt.clf()

    print(total_norms)

    return EarlyStop.early_stop



##############################
#
# VALIDATION FUNCTION
#
##############################

def eval_on_batch(mode, args, input_tensor, target_tensor, target_number_tensor=None, eos_target=None, encoder=None, decoder=None, regressor=None, eos=None, dec_criterion=None, reg_criterion=None, eos_criterion=None, device='cpu'):
    """Possible values for 'mode' arg: {"eval_seq2seq", "eval_reg", "eval_eos", "test"}"""

    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    if mode == "eval_reg":
        regressor_output = regressor(encoder_outputs)
        reg_loss = reg_criterion(regressor_output, target_number_tensor)
        return reg_loss.item()

    elif mode == "eval_seq2seq":
        decoder_input = torch.zeros(1, args.batch_size, args.ind_size, device=decoder.device).to(device)

        #if encoder.hidden_size == decoder.hidden_size:
        if args.enc_dec_hidden_init:
            decoder_hidden_0 = encoder_hidden.expand(decoder.num_layers, args.batch_size, decoder.hidden_size)
        else:
            decoder_hidden_0 = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size, device=device)
        #decoder_hidden_0 = torch.zeros(decoder.num_layers, args.batch_size, decoder.hidden_size, device=device)
        dec_loss = 0
        denom = 0
        l2_distances = []
        total_dist = torch.zeros([args.ind_size], device=device).float()
        norms =[] 
        norm_loss = 0
        #for b in range(args.batch_size):
        for b in range(args.batch_size):
            single_dec_input = decoder_input[:, b].view(1, 1, -1)
            decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
            l_loss = 0
            for l in range(target_number_tensor[b].int()):
                decoder_output, new_decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden.contiguous()) #input_lengths=torch.tensor([target_number_tensor[b]])
                new_norm = torch.norm(decoder_output).item()
                if new_norm > 0.1:
                    norms.append(new_norm)
                decoder_hidden = new_decoder_hidden
                output_norm = torch.norm(decoder_output, dim=-1)
                mean_norm = output_norm.mean()
                norm_loss += F.relu(1-mean_norm)
                
                l_loss += dec_criterion(decoder_output, target_tensor[l, b].unsqueeze(0).unsqueeze(0))
                #single_dec_input = decoder_output
                single_dec_input = target_tensor[l,b].unsqueeze(0).unsqueeze(0)
                denom += 1
                #print(decoder_output.squeeze(), target_tensor[l,b].squeeze())
                l2 = torch.norm(decoder_output.squeeze()-target_tensor[l,b].squeeze(),2).item()
                dist = decoder_output.squeeze()-target_tensor[l,b].squeeze()
                total_dist += dist
                #print('new l2 at', l, b, l2)
                l2_distances.append(l2)
                #print('\tcomputing norm between {} and {}, result: {}'.format(decoder_output.squeeze()[0].item(), target_tensor[l,b].squeeze()[0].item(), l2))
                #print('\tcomputing loss between {} and {}, result: {}'.format(decoder_output.squeeze()[0].item(), target_tensor[l,b].squeeze()[0].item(), l_loss))
            #dec_loss += l_loss/float(l)
            dec_loss += l_loss

        print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        #dec_loss /= float(b) 
        #print('dist', total_dist)
        dec_loss = dec_loss*args.ind_size/torch.sum(target_number_tensor)
        norm_loss /= torch.sum(target_number_tensor)
        if dec_loss < 2.5:
            #print(decoder_output.squeeze())
            #print(target_tensor[l,b].squeeze())
            #print('l2:', torch.norm(decoder_output-target_tensor[l,b],2).item())
            pass
        return dec_loss.item(), norm_loss.item(), norms

    elif mode == "eval_eos":
        eos_input = torch.zeros(1, args.batch_size, args.ind_size, device=eos.device).to(device)
        eos_hidden = encoder_hidden[encoder.num_layers-1:encoder.num_layers]
        eos_inputs = torch.cat((eos_input, target_tensor[:-1]))
        eos_outputs, hidden = eos(input=eos_inputs, input_lengths=target_number_tensor, encoder_outputs=encoder_outputs, hidden=eos_hidden)
        eos_target = eos_target[:eos_outputs.shape[0],:].permute(1,0)
        eos_outputs = eos_outputs.squeeze(2).permute(1,0)
        
        arange = torch.arange(0, eos_outputs.shape[1], step=1).expand(args.batch_size, eos_outputs.shape[1]).cuda()
        lengths = target_number_tensor.expand(eos_outputs.shape[1], args.batch_size).long().cuda()
        lengths = lengths.permute(1,0)
        mask = arange < lengths 
        mask = mask.cuda().float()
        eos_target = torch.argmax(eos_target, dim=1)
        
        eos_loss = eos_criterion(eos_outputs*mask, eos_target*mask)
        return eos_loss.item()


