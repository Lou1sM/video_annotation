import subprocess 
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
from search_threshes import find_best_thresh_from_probs, compute_f1_for_thresh


def get_outputs(encoder, decoder, data_generator, gt_forcing, ind_size, mode='seq2seq', device='cuda'):
    encoder.eval()
    decoder.eval()
    encoder.batch_size = 1
    decoder.batch_size = 1

    norms = []
    output = []
    positions = []
    total_l2_distances = []
    cos_sims = []
    eos_results = []
    nums_of_inds = {}
    tup_sizes_by_pos = {}
    test_info = {}
    
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
        eos_preds = []
        dec_out_list = []
        for l in range(target_number.int()):
            try:
                decoder_output, eos_pred, decoder_hidden_new = decoder(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden) 
                eos_preds.append(eos_pred.item())
            except ValueError:
                decoder_output, decoder_hidden_new = decoder(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden) 
            dec_out_list.append(decoder_output)
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
        try:
            for l in range(target_number.int(), 29):
                decoder_output, eos_pred, decoder_hidden_new = decoder(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden) 
                eos_preds.append(eos_pred.item())
                decoder_hidden = decoder_hidden_new
            #eos_guess = int(np.argmax(eos_preds))
            # Take first element that's greater that 0.5
            eos_guess = [i>0.5 for i in eos_preds].index(True)
            eos_gt = target_number.item()-1
            eos_result = (eos_guess == eos_gt)
            eos_results.append(eos_result)
        except ValueError:
            eos_guess = eos_target = eos_result = -1
            pass
        assert len(dp_output) == len(gt_embeddings), "Wrong number of embeddings in output"
        #print(eos_preds, eos_guess, target_number.item()-1, eos_result)
        output.append({'videoId':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'l2_distances': l2_distances, 'avg_l2_distance': sum(l2_distances)/len(l2_distances), 'eos_guess': eos_guess, 'eos_gt': target_number.item(), 'eos_result': eos_result})
    assert sum(list(nums_of_inds.values())) == len(norms)
    avg_l2_distance = round(sum(l2_distances)/len(l2_distances),4)
    avg_cos_sim = round(sum(cos_sims)/len(cos_sims),4)
    avg_norm = round( (sum([t[0] for t in tup_sizes_by_pos.values()])/sum(nums_of_inds.values())),4)
    eos_accuracy = sum(eos_results)/(len(eos_results)+1e-5)
    test_info['l2_distance'] = avg_l2_distance
    test_info['cos_similarity'] = avg_cos_sim
    test_info['avg_norm'] = avg_norm
    test_info['eos_accuracy'] = eos_accuracy
    #print('avg_l2_distance', sum(l2_distances)/len(l2_distances))
    #print('avg_l2_distance', avg_l2_distance)
    #print('avg_cos_sim', avg_cos_sim)
    #print('total number of embeddings:', sum(nums_of_inds.values()))
    #print('eos_accuracy:', eos_accuracy)
    #print('number of embeddings at each position:')
    for k,v in nums_of_inds.items():
        print(k,v)
    sizes_by_pos = {k: v[0]/v[1] for k,v in tup_sizes_by_pos.items()}
    for k,v in sizes_by_pos.items():
        print(k,v)
    return output, norms, positions, sizes_by_pos, test_info


def write_outputs_get_info(encoder, decoder, ARGS, data_generator, gt_forcing, exp_name, dset_fragment, fixed_thresh=None):

    outputs, norms, positions, sizes_by_pos, test_info  = get_outputs(encoder, decoder, gt_forcing=gt_forcing ,data_generator=data_generator, ind_size=ARGS.ind_size, device=ARGS.device)

    plt.xlim(0,2)
    bins = np.arange(0,2,.01)
    plt.hist(norms, bins=bins)
    plt.xlabel('l2 norm')
    plt.ylabel('number of embeddings')
    plt.title("{}: Embedding Norms".format(exp_name))
    plt.savefig('../experiments/{}/{}-{}norms-histogram.png'.format(exp_name, exp_name, dset_fragment))
    plt.clf()
    
    plt.xlim(0,10)
    plt.scatter(x=positions, y=norms)
    plt.xlabel('position')
    plt.ylabel('embedding norm')
    plt.title("{}: Norms by Position".format(exp_name))
    plt.savefig('../experiments/{}/{}-{}_norms_position_scatter.png'.format(exp_name, exp_name, dset_fragment))
    plt.clf()
  
    plt.xlim(0,29)
    plt.ylim(0.0, 1.35)
    plt.plot(list(sizes_by_pos.values()))
    plt.xlabel('position')
    plt.ylabel('avg embedding norm')
    plt.title("{}: Avg Norms by Position".format(exp_name))
    plt.savefig('../experiments/{}/{}-{}_avg_norms_position_trend.png'.format(exp_name, exp_name, dset_fragment))
    plt.clf()

    outputs_filename = '../experiments/{}/{}-{}_outputs.txt'.format(exp_name, exp_name, dset_fragment)
    print('writing outputs to', outputs_filename)
    with open(outputs_filename, 'w') as outputs_file:
        json.dump(outputs, outputs_file)
   
    metric_data, total_metric_data, positive_probs, negative_probs = find_best_thresh_from_probs(exp_name, dset_fragment)
    test_info.update(metric_data)
    legit_thresh = fixed_thresh if dset_fragment == 'test' else metric_data['thresh'] 
    print(fixed_thresh, metric_data['thresh'])
    test_info['legit_f1'] = compute_f1_for_thresh(positive_probs, negative_probs, legit_thresh)[6]
    
    plt.xlim(0,len(total_metric_data['thresh']))
    plt.ylim(0.0, max(total_metric_data['f1']) + 0.1)
    plt.plot(total_metric_data['f1'], label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title("{}: F1 scores by threshold".format(exp_name))
    plt.savefig('../experiments/{}/{}-{}_F1_scores_by_threshold.png'.format(exp_name, exp_name, dset_fragment))
    plt.clf()

    #assert test_info['pat_norm'] - test_info['avg_norm'] < .04
    #assert test_info['pat_distance'] - test_info['l2_distance'] < .04
    print(test_info)
    return sizes_by_pos, test_info
 
if __name__=="__main__":

    device='cuda'
    parser = argparse.ArgumentParser()
    #parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--gt_forcing", "-gtf",  action="store_true")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--enc_zeroes", action="store_true")
    parser.add_argument("--dec_zeroes", action="store_true")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--ind_size", type=int, default=10)

    ARGS = parser.parse_args() 
    ARGS.device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = '../checkpoints/{}.pt'.format(ARGS.exp_name)
    gtf = ARGS.gt_forcing
    print("Ground Truth Forcing:", gtf)
    print('enc_zeroes', ARGS.enc_zeroes)
    print('dec_zeroes', ARGS.dec_zeroes)
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    train_norms = []
    val_norms = []
    test_norms = []

    print('getting outputs and info for val set')
    val_lookup_table = video_lookup_table_from_range(1201, 1301, cnn="vgg")
    val_data_generator = load_data_lookup('../data/rdf_video_captions/{}d-val.h5'.format(ARGS.ind_size), video_lookup_table=val_lookup_table, batch_size=1, shuffle=False)
    _, val_info = write_outputs_get_info(ARGS=ARGS, encoder=encoder, decoder=decoder, data_generator=val_data_generator, gt_forcing=gtf, exp_name=ARGS.exp_name, dset_fragment='val')


    if not ARGS.quick:
        print('getting outputs and info for train set')
        train_lookup_table = video_lookup_table_from_range(1, 1201, cnn="vgg")
        train_data_generator = load_data_lookup('../data/rdf_video_captions/{}d-train.h5'.format(ARGS.ind_size), video_lookup_table=train_lookup_table, batch_size=1, shuffle=False)
        _, train_info = write_outputs_get_info(ARGS=ARGS, encoder=encoder, decoder=decoder, data_generator=train_data_generator, gt_forcing=gtf, exp_name=ARGS.exp_name, dset_fragment='train')


        print('getting outputs and info for test set')
        fixed_thresh = ((train_info['thresh']*1200)+(val_info['thresh']*100))/1300
        print('outer_fixed_thresh', fixed_thresh)
        test_lookup_table = video_lookup_table_from_range(1301, 1971, cnn="vgg")
        test_data_generator = load_data_lookup('../data/rdf_video_captions/{}d-test.h5'.format(ARGS.ind_size), video_lookup_table=test_lookup_table, batch_size=1, shuffle=False)
        _, test_info = write_outputs_get_info(ARGS=ARGS, encoder=encoder, decoder=decoder, data_generator=test_data_generator, gt_forcing=gtf, exp_name=ARGS.exp_name, dset_fragment='test', fixed_thresh=fixed_thresh)
        print('TEST')
        print(test_info)

