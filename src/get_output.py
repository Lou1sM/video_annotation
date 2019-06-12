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


def get_output(encoder, decoder, data_generator, teacher_forcing, ind_size, mode='seq2seq', device='cuda'):

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
        encoder_hidden = encoder.initHidden()
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.zeros(1, 1, ind_size, device=decoder.device).to(device)
        if encoder.num_layers == decoder.num_layers and False:
            decoder_hidden = encoder_hidden
        else:
            decoder_hidden = decoder.initHidden()
        dp_output = []
        gt_embeddings = []
        l2_distances = []
        l_loss = 0
        for l in range(target_number.int()):
            decoder_output, decoder_hidden_new = decoder(input=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden) 
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
            #l2 = torch.norm(decoder_output.squeeze()-target_tensor[l,b].squeeze(),2).item()
            #l2_distances.append(l2.squeeze().detach().cpu().numpy().tolist())
            l2_distances.append(l2)
            l_loss += loss
        assert len(dp_output) == len(gt_embeddings), "Wrong number of embeddings in output"
        output.append({'videoId':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'l2_distances': l2_distances, 'avg_l2_distance': sum(l2_distances)/len(l2_distances)})
    print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
    return output


if __name__=="__main__":

    device='cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--teacher_forcing", "-tf",  action="store_true")
    parse.add_argument("--test_name", type=str, default="")

    ARGS = parser.parse_args() 
    checkpoint_path = ARGS.checkpoint_path
    tf = ARGS.teacher_forcing
    print("Teacher forcing:", tf)
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    print('loading lookup tables..\n')
    train_lookup_table = video_lookup_table_from_range(1,1201, cnn="vgg")
    val_lookup_table = video_lookup_table_from_range(1201, 1301, cnn="vgg")
    test_lookup_table = video_lookup_table_from_range(1301, 1971, cnn="vgg")

    train_data_generator = load_data_lookup('../data/rdf_video_captions/train_10d-det.h5', video_lookup_table=train_lookup_table, batch_size=1, shuffle=False)
    val_data_generator = load_data_lookup('../data/rdf_video_captions/val_10d-det.h5', video_lookup_table=val_lookup_table, batch_size=1, shuffle=False)
    test_data_generator = load_data_lookup('../data/rdf_video_captions/test_10d-det.h5', video_lookup_table=test_lookup_table, batch_size=1, shuffle=False)
    
    print('Getting train outputs')
    train_output = get_output(encoder, decoder, teacher_forcing=tf, ind_size=10, data_generator=train_data_generator)
    print('Getting val outputs')
    val_output = get_output(encoder, decoder, teacher_forcing=tf, ind_size=10, data_generator=val_data_generator)
    print('Getting test outputs')
    test_output = get_output(encoder, decoder, teacher_forcing=tf, ind_size=10, data_generator=test_data_generator)

    if tf:
        train_filename = 'output_10d-det-norm1-tf-train-rerun{}.txt'.format(args.test_name)
        val_filename = 'output_10d-det-norm1-tf-val-rerun{}.txt'.format(args.test_name)
        test_filename = 'output_10d-det-norm1-tf-test-rerun{}.txt'.format(args.test_name)
    else:
        train_filename = 'output_10d-det-norm1-notf-train-rerun{}.txt'.format(args.test_name)
        val_filename = 'output_10d-det-norm1-notf-val-rerun{}.txt'.format(args.test_name)
        test_filename = 'output_10d-det-norm1-notf-test-rerun{}.txt'.format(args.test_name)

    with open(train_filename, 'w') as train_outfile:
        json.dump(train_output, train_outfile)
    
    with open(val_filename, 'w') as val_outfile:
        json.dump(val_output, val_outfile)
 
    with open(test_filename, 'w') as test_outfile:
        json.dump(val_output, test_outfile)
