import argparse
import torch.nn as nn
import sys
import os
import json
import sys
import utils
import options
import models
import torch
import numpy as np
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids


def get_output(checkpoint_path, input_tensor, target_number_tensor, rnge, mode='seq2seq', device='cuda'):

    #Load trained model 
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    print(encoder)
    print(decoder)

    # if mode == 'eos':
    #   num_predictor = checkpoint['eos']
    # elif mode == 'regressor':
    #   num_predictor = checkpoint['regressor']
    # else:
    #   print("Wrong input for mode parameter")
    #   exit()

    num_test_samples = 3
    encoder.batch_size = num_test_samples
    decoder.batch_size = num_test_samples
    #num_predictor.batch_size = num_test_samples

    # Pass input through encoder
    encoder_hidden = encoder.initHidden().to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.zeros(1, decoder.batch_size, decoder.hidden_size, device=decoder.device).to(device)
    decoder_hidden_0 = encoder_hidden[encoder.num_layers-1:encoder.num_layers]

    print("real number of embeddings in video: ", target_number_tensor)
        
    batch_decoder_output = []
    for b in range(num_test_samples):
        single_dec_input = decoder_input[:, b].view(1, 1, -1)
        decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
        single_dec_output = []
        for l in range(target_number_tensor[b].int()):
            decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden) #input_lengths=torch.tensor([target_number_tensor[b]])
            print("decoder output shape:", decoder_output.shape)
            single_dec_output.append(decoder_output.squeeze().detach().cpu().numpy().tolist())
            single_dec_input = decoder_output
        print("number of embeddings in video:", len(single_dec_output))
        batch_decoder_output.append({'videoId':str(rnge[0]+b), 'embeddings':single_dec_output})
    return batch_decoder_output



#def get_output_gen(checkpoint_path, data_generator, mode='seq2seq', device='cuda'):
def get_output_gen(encoder, decoder, data_generator, teacher_forcing, ind_size, mode='seq2seq', device='cuda'):

    encoder.eval()
    decoder.eval()
    encoder.batch_size = 1
    decoder.batch_size = 1

    output = []
    # Pass input through encoder
    for iter_, dp in enumerate(data_generator):
        input_tensor = dp[0].float().transpose(0,1).to(device)
        target_tensor = dp[1].float().transpose(0,1).to(device)
        target_number = dp[2].float().to(device)
        video_id = dp[4].item()
        #print('video_id', video_id)
        #print('first inp elem', input_tensor[0,0,0,0,0].item())
        #print('first outp elem', target_tensor[0,0,0].item())
        #print(input_tensor.shape)

        #encoder_hidden = encoder.initHidden().to(device)
        encoder_hidden = encoder.initHidden()
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        #print("real number of embeddings in video: ", int(target_number.item()))
        decoder_input = torch.zeros(1, 1, ind_size, device=decoder.device).to(device)
        if encoder.num_layers == decoder.num_layers and False:
            decoder_hidden = encoder_hidden
        else:
            #decoder_hidden = torch.zeros(decoder.num_layers, 1, decoder.hidden_size).to(device)
            decoder_hidden = decoder.initHidden()
        dp_output = []
        gt_embeddings = []
        l2_distances = []
        l_loss = 0
        for l in range(target_number.int()):
            decoder_output, decoder_hidden_new = decoder(input=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden) 
            #print(decoder_input.squeeze()[0].item(), encoder_outputs.squeeze()[0,0].item(), decoder_hidden.squeeze()[0,0].item(), 'GOES TO', decoder_output.squeeze()[0].item())
            decoder_hidden = decoder_hidden_new
            dp_output.append(decoder_output.squeeze().detach().cpu().numpy().tolist())
            gt_embeddings.append(target_tensor[l].squeeze().detach().cpu().numpy().tolist())
            if teacher_forcing:
                decoder_input = target_tensor[l].unsqueeze(0)
            else:
                decoder_input = decoder_output
            criterion = nn.MSELoss()
            loss = criterion(decoder_output.squeeze(), target_tensor[l].squeeze())
            l2 = torch.norm(decoder_output.squeeze() - target_tensor[l].squeeze(), 2).item()
            #print('\tcomputing norm between {} and {}, result: {}'.format(decoder_output.squeeze()[0].item(), target_tensor[l].squeeze()[0].item(), l2))
            #l2 = torch.norm(decoder_output.squeeze()-target_tensor[l,b].squeeze(),2).item()
            #print('new l2 in output at', l, l2)
            #l2_distances.append(l2.squeeze().detach().cpu().numpy().tolist())
            l2_distances.append(l2)
            l_loss += loss
            #print(decoder_output)
            #print(target_tensor[l].squeeze())

        #print('final_loss:', (l_loss*50/target_number).item())
        #print("number of embeddings in video:", len(dp_output))
        #print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        #print(target_tensor[0,0])
        assert len(dp_output) == len(gt_embeddings), "Wrong number of embeddings in output"
        output.append({'videoId':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'l2_distances': l2_distances, 'avg_l2_distance': sum(l2_distances)/len(l2_distances)})
    return output


if __name__=="__main__":

    device='cuda'
    #checkpoint_path = sys.argv[1]
    #tf = sys.argv[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--teacher_forcing", "-tf",  action="store_true")

    ARGS = parser.parse_args() 
    checkpoint_path = ARGS.checkpoint_path
    tf = ARGS.teacher_forcing
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    print('loading lookup tables..')
    #lookup_table = video_lookup_table_from_range(1,4, cnn="vgg_old")
    train_lookup_table = video_lookup_table_from_range(1,1201, cnn="vgg")
    val_lookup_table = video_lookup_table_from_range(1201, 1301, cnn="vgg")
    #lookup_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874])

    train_data_generator = load_data_lookup('../data/rdf_video_captions/train_10d-det.h5', video_lookup_table=train_lookup_table, batch_size=1, shuffle=False)
    val_data_generator = load_data_lookup('../data/rdf_video_captions/val_10d-det.h5', video_lookup_table=val_lookup_table, batch_size=1, shuffle=False)
    #output = get_output_gen(checkpoint_path, data_generator=test_data_generator)
    train_output = get_output_gen(encoder, decoder, teacher_forcing=tf, ind_size=10, data_generator=train_data_generator)
    val_output = get_output_gen(encoder, decoder, teacher_forcing=tf, ind_size=10, data_generator=val_data_generator)

    #print(output)
    if tf:
        train_filename = 'output_10d-det-norm1-tf-train-rerun.txt'
        val_filename = 'output_10d-det-norm1-tf-val-rerun.txt'
    else:
        train_filename = 'output_10d-det-norm1-notf-train-rerun.txt'
        val_filename = 'output_10d-det-norm1-notf-val-rerun.txt'

    with open(train_filename, 'w') as train_outfile:
        json.dump(train_output, train_outfile)
    
    with open(val_filename, 'w') as val_outfile:
        json.dump(val_output, val_outfile)
