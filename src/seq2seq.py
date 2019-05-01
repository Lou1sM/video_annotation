import random
import re
import string
import unicodedata

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class EncoderRNN(nn.Module):
    def __init__(self, args, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = args.enc_size
        self.device = device
                
        #Use VGG, NOTE: param.requires_grad are set to True by default
        self.vgg = models.vgg19(pretrained=True)
        num_ftrs = self.vgg.classifier[6].in_features
        #Change the size of VGG output to the GRU size
        self.vgg.classifier[6] = nn.Linear(num_ftrs, args.enc_size)

        self.gru = nn.GRU(args.enc_size, args.enc_size)

    def forward(self, input, hidden):
        embedded = self.vgg(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, args, device):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = args.ind_size
        self.output_size = args.ind_size
        self.dropout_p = args.dropout
        self.max_length = args.max_length

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
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
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


def train(args, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    teacher_forcing_ratio = args.teacher_forcing_ratio

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(args.max_length, encoder.hidden_size, device=encoder.device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    #decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.zeros(args.ind_size, device=decoder.device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    #REMOVE WHEN FIGURED OUT HOW TO PREDICT NUMBER!!!
    use_teacher_forcing = True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
 
    #TO DO: see how to predict the number of tokens 
    # else:
    #     # Without teacher forcing: use its own predictions as the next input
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs)
    #         topv, topi = decoder_output.topk(1)
    #         decoder_input = topi.squeeze().detach()  # detach from history as input

    #         loss += criterion(decoder_output, target_tensor[di])
    #         if decoder_input.item() == EOS_token:
    #             break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def tensorsFromPair(pairs, args):
	# given a couple video + set of embeddings of all individuals it returns a couple:
	# (input_tensor, output_tensor)
	# where:
	# - input_tensor (num_frames, 1, 3, 224, 224)
	# - output_tensor (num_individuals, embedding_size)
	return (torch.rand(args.num_frames, 1, 3, 224, 224), torch.rand(args.max_length, args.ind_size))


def trainIters(args, encoder, decoder, print_every=1000, plot_every=100):

    learning_rate = args.learning_rate
    n_iters = args.max_epochs

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    pairs = "dummy"

    training_pairs = [tensorsFromPair(random.choice(pairs), args)
                      for i in range(n_iters)]
    criterion = nn.MSELoss()

    #TO DO: add early stopping
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(args, input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

