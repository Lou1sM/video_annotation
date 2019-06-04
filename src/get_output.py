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



def get_output_gen(checkpoint_path, data_generator, mode='seq2seq', device='cuda'):
    #Load trained model 
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

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
        print('video_id', video_id)
        print('first inp elem', input_tensor[0,0,0,0,0].item())
        print('first outp elem', target_tensor[0,0,0].item())
        print(input_tensor.shape)

        #encoder_hidden = encoder.initHidden().to(device)
        encoder_hidden = encoder.initHidden()
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        #print("real number of embeddings in video: ", int(target_number.item()))
        decoder_input = torch.zeros(1, 1, 50, device=decoder.device).to(device)
        if encoder.num_layers == decoder.num_layers and False:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = torch.zeros(decoder.num_layers, 1, decoder.hidden_size).to(device)
        dp_output = []
        gt_embeddings = []
        l2_distances = []
        l_loss = 0
        for l in range(target_number.int()):
            decoder_output, decoder_hidden_new = decoder(input=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden) 
            print(decoder_input.squeeze()[0].item(), encoder_outputs.squeeze()[0,0].item(), decoder_hidden.squeeze()[0,0].item(), 'GOES TO', decoder_output.squeeze()[0].item())
            decoder_hidden = decoder_hidden_new
            dp_output.append(decoder_output.squeeze().detach().cpu().numpy().tolist())
            gt_embeddings.append(target_tensor[iter_].squeeze().detach().cpu().numpy().tolist())
            decoder_input = decoder_output.view(1,1,-1)
            criterion = nn.MSELoss()
            #print(decoder_output[0].shape, target_tensor[iter_].shape)
            loss = criterion(decoder_output.squeeze(), target_tensor[iter_].squeeze())
            #print(loss.item())
            #print(decoder_output.squeeze(), target_tensor[iter_].squeeze())
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
        print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
        #print(target_tensor[0,0])
        output.append({'videoId':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'l2_distances': l2_distances, 'avg_l2_distance': sum(l2_distances)/len(l2_distances)})
    return output


if __name__=="__main__":

    device='cuda'
    #checkpoint_path = '/home/eleonora/video_annotation/checkpoints/chkpt_batch3_lr0.001_enc1_dec1_tfratio1.0_wgDecay0.0_Adam.pt'
    #checkpoint_path = '/home/louis/video_annotation/checkpoints/chkpt_batch3_lr0.001_enc1_dec1_tfratio1.0_wgDecay0.0_Adam.pt'
    #checkpoint_path = '../checkpoints/chkpt05-26_16:29:28.pt'
    checkpoint_path = sys.argv[1]

    lookup_table = video_lookup_table_from_range(1,4, cnn="vgg_old")
    #lookup_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874])

    #test_data_generator = load_data_lookup('../data/rdf_video_captions/50d.6dp.h5', video_lookup_table=lookup_table, batch_size=1, shuffle=False)
    test_data_generator = load_data_lookup('/home/eleonora/video_annotation/data/rdf_video_captions/50d_overfitting.h5', video_lookup_table=lookup_table, batch_size=1, shuffle=False)
    """
    for iter_, training_triplet in enumerate(h5_test_generator):
        input_tensor = training_triplet[0].float().transpose(0,1).to(device)
        target_tensor = training_triplet[1].float().transpose(0,1).to(device)
        target_number = training_triplet[2].float().to(device)

    output = get_output(checkpoint_path, input_tensor, target_number, rnge)
    #print(output)
    """
    output = get_output_gen(checkpoint_path, data_generator=test_data_generator)

    #print(output)
    with open('output.txt', 'w') as outfile:
        json.dump(output, outfile)
