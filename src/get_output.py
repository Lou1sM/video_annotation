import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
import torch.nn as nn
import sys
import os
import json
import sys
import utils
import options
import torch
import numpy as np
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids


def get_outputs(encoder, decoder, enc_zeroes, dec_zeroes, data_generator, gt_forcing, ind_size, mode='seq2seq', device='cuda'):
    encoder.eval()
    decoder.eval()
    encoder.batch_size = 1
    decoder.batch_size = 1

    norms = []
    output = []
    positions = []
    total_l2_distances = []
    cos_sims = []
    nums_of_inds = {}
    tup_sizes_by_pos = {}
    test_info = {}
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
            new_norm = torch.norm(decoder_output).item()
            norms.append(new_norm)
            try:
                nums_of_inds[l] += 1
            except KeyError:
                nums_of_inds[l] = 1

            try:
                old_num, old_denom = tup_sizes_by_pos[l]
                tup_sizes_by_pos[l] = old_num+new_norm, old_denom+1
            except KeyError:
                tup_sizes_by_pos[l] = new_norm, 1
            
            if gt_forcing:
                decoder_input = target_tensor[l].unsqueeze(0)
            else:
                decoder_input = decoder_output
            criterion = nn.MSELoss()
            loss = criterion(decoder_output.squeeze(), target_tensor[l].squeeze())
            l_loss += loss
            l2 = torch.norm(decoder_output.squeeze() - target_tensor[l].squeeze(), 2).item()
            l2_distances.append(l2)
            cos_sim = F.cosine_similarity(decoder_output.squeeze(), target_tensor[l].squeeze(),0).item()
            cos_sims.append(cos_sim)
            positions.append(l+1)
        assert len(dp_output) == len(gt_embeddings), "Wrong number of embeddings in output"
        output.append({'videoId':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'l2_distances': l2_distances, 'avg_l2_distance': sum(l2_distances)/len(l2_distances)})
    assert sum(list(nums_of_inds.values())) == len(norms)
    avg_l2_distance = round(sum(l2_distances)/len(l2_distances),4)
    avg_cos_sim = round(sum(cos_sims)/len(cos_sims),4)
    avg_emb_norm = round( (sum([t[0] for t in tup_sizes_by_pos.values()])/sum(nums_of_inds.values())),4)
    test_info['l2_distance'] = avg_l2_distance
    test_info['cos_similarity'] = avg_cos_sim
    test_info['avg_emb_norm'] = avg_emb_norm
    #print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
    print('avg_l2_distance', avg_l2_distance)
    print('avg_cos_sim', avg_cos_sim)
    print('total number of embeddings:', sum(nums_of_inds.values()))
    print('number of embeddings at each position:')
    for k,v in nums_of_inds.items():
        print(k,v)
    sizes_by_pos = {k: v[0]/v[1] for k,v in tup_sizes_by_pos.items()}
    for k,v in sizes_by_pos.items():
        print(k,v)
    return output, norms, positions, sizes_by_pos, test_info


def write_outputs_get_info(encoder, decoder, ARGS, data_generator, gt_forcing, exp_name, test_name):

    outputs, norms, positions, sizes_by_pos, test_info  = get_outputs(encoder, decoder, enc_zeroes=ARGS.enc_zeroes, dec_zeroes=ARGS.dec_zeroes, gt_forcing=gt_forcing ,data_generator=data_generator, ind_size=ARGS.ind_size, device=ARGS.device)

    plt.xlim(0,2)
    bins = np.arange(0,2,.01)
    plt.hist(norms, bins=bins)
    plt.xlabel('l2 norm')
    plt.ylabel('number of embeddings')
    plt.title("{}: Embedding Norms".format(test_name))
    plt.savefig('../experiments/{}/{}norms-histogram.png'.format(exp_name, test_name))
    plt.clf()
    
    plt.xlim(0,10)
    plt.scatter(x=positions, y=norms)
    plt.xlabel('position')
    plt.ylabel('embedding norm')
    plt.title("{}: Norms by Position".format(test_name))
    plt.savefig('../experiments/{}/{}_norms_position_scatter.png'.format(exp_name, test_name))
    plt.clf()
  
    plt.xlim(0,29)
    plt.ylim(0.0, 1.35)
    plt.plot(list(sizes_by_pos.values()))
    plt.xlabel('position')
    plt.ylabel('avg embedding norm')
    plt.title("{}: Avg Norms by Position".format(test_name))
    plt.savefig('../experiments/{}/{}_avg_norms_position_trend.png'.format(exp_name, test_name))
    plt.clf()

  
    outputs_filename = '../experiments/{}/{}_outputs.txt'.format(exp_name, test_name)
    with open(outputs_filename, 'w') as outputs_file:
        json.dump(outputs, outputs_file)
   
    return sizes_by_pos, test_info
 
if __name__=="__main__":

    device='cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--gt_forcing", "-gtf",  action="store_true")
    parser.add_argument("--test_name", type=str, default="")
    parser.add_argument("--enc_zeroes", action="store_true")
    parser.add_argument("--dec_zeroes", action="store_true")

    ARGS = parser.parse_args() 
    checkpoint_path = ARGS.checkpoint_path
    gtf = ARGS.gt_forcing
    print("Ground Truth Forcing:", gtf)
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    train_norms = []
    val_norms = []
    test_norms = []

    print('getting outputs and info for val set')
    val_lookup_table = video_lookup_table_from_range(1201, 1301, cnn="vgg")
    val_data_generator = load_data_lookup('../data/rdf_video_captions/val_10d-det.h5', video_lookup_table=val_lookup_table, batch_size=1, shuffle=False)
    get_outputs_and_info(encoder=encoder, decoder=decoder, enc_zeroes=ARGS.enc_zeroes, dec_zeroes=ARGS.dec_zeroes, data_generator=val_data_generator, ind_size=10, device='cuda', gt_forcing=gtf, test_name='val_{}'.format(ARGS.test_name))


    print('getting outputs and info for train set')
    train_lookup_table = video_lookup_table_from_range(1, 1201, cnn="vgg")
    train_data_generator = load_data_lookup('../data/rdf_video_captions/train_10d-det.h5', video_lookup_table=train_lookup_table, batch_size=1, shuffle=False)
    get_outputs_and_info(encoder=encoder, decoder=decoder, enc_zeroes=ARGS.enc_zeroes, dec_zeroes=ARGS.dec_zeroes, data_generator=train_data_generator, ind_size=10, device='cuda', gt_forcing=gtf, test_name='train_{}'.format(ARGS.test_name))


    print('getting outputs and info for test set')
    test_lookup_table = video_lookup_table_from_range(1301, 1971, cnn="vgg")
    test_data_generator = load_data_lookup('../data/rdf_video_captions/test_10d-det.h5', video_lookup_table=test_lookup_table, batch_size=1, shuffle=False)
    get_outputs_and_info(encoder=encoder, decoder=decoder, enc_zeroes=ARGS.enc_zeroes, dec_zeroes=ARGS.dec_zeroes, data_generator=test_data_generator, ind_size=10, device='cuda', gt_forcing=gtf, test_name='test_{}'.format(ARGS.test_name))

