import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse
import torch.nn as nn
import os
import json
import torch
from torchvision import models
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids
from search_threshes import find_best_thresh_from_probs, compute_scores_for_thresh


def get_outputs(encoder, decoder, transformer, data_generator, ind_size, setting='embeddings', device='cuda'):
    encoder.eval()
    decoder.eval()
    encoder.batch_size = 1
    decoder.batch_size = 1
                
    criterion = nn.MSELoss()

    output = []
    positions = []
    total_l2_distances = []
    total_cos_sims = []
    test_info = {}
    
    for iter_, dp in enumerate(data_generator):
        input_tensor = dp[0].float().transpose(0,1).to(device)
        target_tensor = dp[1].float().transpose(0,1).to(device)
        target_number = dp[2].float().to(device).int()
        video_id = dp[3].item()
        
        encoder_hidden = encoder.initHidden()
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.zeros(1, 1, ind_size, device=decoder.device).to(device)
        decoder_hidden = decoder.initHidden()
        dp_output = []
        gt_embeddings = []
        l_loss = 0
        dec_out_list = []
        for l in range(target_number):
            decoder_output, decoder_hidden_new = decoder(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden)
            decoder_output = F.normalize(decoder_output, p=2, dim=-1)
            dec_out_list.append(decoder_output)
            decoder_hidden = decoder_hidden_new
            dp_output.append(decoder_output.squeeze().detach().cpu().numpy().tolist())
            gt_embeddings.append(target_tensor[l].squeeze().detach().cpu().numpy().tolist())

            emb_pred = decoder_output
            decoder_input = emb_pred
            loss = criterion(emb_pred.squeeze(), target_tensor[l].squeeze())
            l_loss += loss
            l2 = torch.norm(emb_pred.squeeze() - target_tensor[l].squeeze(), 2).item()
            assert abs(torch.norm(emb_pred.squeeze(), 2).item() - 1) < 1e-3, "{} should be 1".format(torch.norm(emb_pred.squeeze(), 2).item() )
            assert abs(torch.norm(target_tensor[l].squeeze(), 2).item()) - 1 < 1e-3, "{} should be 1".format(torch.norm(target_tensor[l].squeeze(), 2).item() )
            total_l2_distances.append(l2)
            cos_sim = F.cosine_similarity(emb_pred.squeeze(), target_tensor[l].squeeze(),0).item()
            total_cos_sims.append(cos_sim)
            positions.append(l+1)
        assert len(dp_output) == len(gt_embeddings), "Wrong number of embeddings in output: {} being output but {} in ground truth".format(len(dp_output), len(gt_embeddings))
        output.append({'video_id':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'avg_l2_distance': sum(total_l2_distances)/len(total_l2_distances)}) 
  
    avg_l2_distance = round(sum(total_l2_distances)/len(total_l2_distances),4)
    avg_cos_sim = round(sum(total_cos_sims)/len(total_cos_sims),4)
    assert len(total_l2_distances) == len(total_cos_sims), "l2: {}, cos_sim: {}".format(len(total_l2_distances), len(total_cos_sims))
    test_info['l2_distance'] = avg_l2_distance
    test_info['cos_similarity'] = avg_cos_sim
    print(test_info)
    return output, positions, test_info


def write_outputs_get_info(encoder, decoder, transformer, ARGS, dataset, data_generator, exp_name, dset_fragment, fixed_thresh=None, setting='embeddings'):

    outputs, positions, test_info  = get_outputs(encoder, decoder, transformer, data_generator=data_generator, ind_size=ARGS.ind_size, device=ARGS.device, setting=setting)

    outputs_filename = '../experiments/{}/{}-{}_outputs.txt'.format(exp_name, exp_name, dset_fragment)
    print('writing outputs to', outputs_filename)
    with open(outputs_filename, 'w') as outputs_file:
        json.dump(outputs, outputs_file)
   
    gt_file_path = f'/data1/louis/data/rdf_video_captions/{dataset}.json'
    with open(gt_file_path) as f:
        gt_json = json.load(f)
        gt_json = {g['video_id']: g for g in gt_json}

    mlp_dict = {}
    weight_dict = torch.load(f"/data1/louis/data/{dataset}-mlps.pickle")
    for relation, weights in weight_dict.items():
        hidden_layer = nn.Linear(weights["hidden_weights"].shape[0], weights["hidden_bias"].shape[0])
        hidden_layer.weight = nn.Parameter(torch.FloatTensor(weights["hidden_weights"]), requires_grad=False)
        hidden_layer.bias = nn.Parameter(torch.FloatTensor(weights["hidden_bias"]), requires_grad=False)
        output_layer = nn.Linear(weights["output_weights"].shape[0], weights["output_bias"].shape[0])
        output_layer.weight = nn.Parameter(torch.FloatTensor(weights["output_weights"]), requires_grad=False)
        output_layer.bias = nn.Parameter(torch.FloatTensor(weights["output_bias"]), requires_grad=False)
        mlp_dict[relation] = nn.Sequential(hidden_layer, nn.ReLU(), output_layer, nn.Sigmoid()) 

    gt_file_path = f"/data1/louis/data/rdf_video_captions/{dataset}.json"

    with open(gt_file_path, 'r') as gt_file:
        gt = json.load(gt_file)

    metric_data, total_metric_data, positive_probs, negative_probs, error_dict = find_best_thresh_from_probs(exp_name, dset_fragment, ind_size=ARGS.ind_size, dataset=dataset, mlp_dict=mlp_dict, gt_json=gt_json)
    test_info.update(metric_data)
    legit_thresh = fixed_thresh if dset_fragment == 'test' else metric_data['thresh'] 
    print(fixed_thresh, metric_data['thresh'])
    test_info['legit_f1'] = compute_scores_for_thresh(positive_probs, negative_probs, legit_thresh)[6]
    
    with open('../experiments/{}/{}-{}errors.json'.format(exp_name, exp_name, dset_fragment), 'w') as error_file:
        json.dump(error_dict, error_file)
    
    plt.xlim(0,len(total_metric_data['thresh']))
    plt.ylim(0.0, max(total_metric_data['f1']) + 0.1)
    plt.plot(total_metric_data['f1'], label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title("{}: F1 scores by threshold".format(exp_name))
    plt.savefig('../experiments/{}/{}-{}_F1_scores_by_threshold.png'.format(exp_name, exp_name, dset_fragment))
    plt.clf()

    print(test_info)
    return test_info
 
if __name__=="__main__":

    device='cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--enc_zeroes", action="store_true")
    parser.add_argument("--dec_zeroes", action="store_true")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--ind_size", type=int, default=10)
    parser.add_argument("--setting", type=str, default='embeddings')
    parser.add_argument("--checkpoint_to_test", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ontology", type=str, default='')


    ARGS = parser.parse_args() 
    ARGS.device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = ARGS.checkpoint_to_test
        
    print("Ground Truth Forcing:", gtf)
    print('enc_zeroes', ARGS.enc_zeroes)
    print('dec_zeroes', ARGS.dec_zeroes)
    checkpoint = torch.load(checkpoint_path, map_location=ARGS.device)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder.to(ARGS.device)
    decoder.to(ARGS.device)
    encoder.device = ARGS.device
    decoder.device = ARGS.device
    try:
        transformer = checkpoint['transformer']
    except KeyError:
        transformer = None

    exp_dir = '../experiments/{}'.format(ARGS.exp_name)
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    print('getting outputs and info for val set')
    dataset = f'{ARGS.dataset}-{ARGS.ontology}-{ARGS.ind_size}d'
    val_lookup_table = video_lookup_table_from_range(1201,1301, dataset='MSVD') if ARGS.dataset == 'MSVD' else video_lookup_table_from_range(6513,7010, dataset='MSRVTT')
    val_data_generator = load_data_lookup(f'/data1/louis/data/rdf_video_captions/{dataset}-val.h5', video_lookup_table=val_lookup_table, batch_size=1, shuffle=False)
    _, val_info = write_outputs_get_info(ARGS=ARGS, dataset=dataset, encoder=encoder, decoder=decoder, transformer=transformer, data_generator=val_data_generator, exp_name=ARGS.exp_name, dset_fragment='val', setting=ARGS.setting)

    if not ARGS.quick:
        print('getting outputs and info for train set')
        train_lookup_table = video_lookup_table_from_range(1,1201, dataset='MSVD')if ARGS.dataset == 'MSVD' else video_lookup_table_from_range(0,6513, dataset='MSRVTT')
        train_data_generator = load_data_lookup(f'/data1/louis/data/rdf_video_captions/{dataset}-train.h5', video_lookup_table=train_lookup_table, batch_size=1, shuffle=False)
        _, train_info = write_outputs_get_info(ARGS=ARGS, dataset=dataset, encoder=encoder, decoder=decoder, transformer=transformer, data_generator=train_data_generator, exp_name=ARGS.exp_name, dset_fragment='train', setting=ARGS.setting)

        print('getting outputs and info for test set')
        fixed_thresh = ((train_info['thresh']*1200)+(val_info['thresh']*100))/1300
        print('outer_fixed_thresh', fixed_thresh)
        test_lookup_table = video_lookup_table_from_range(1301,1971, dataset='MSVD') if ARGS.dataset == 'MSVD' else video_lookup_table_from_range(7010,10000, dataset='MSRVTT')
        test_data_generator = load_data_lookup(f'/data1/louis/data/rdf_video_captions/{dataset}-test.h5')
        _, test_info = write_outputs_get_info(ARGS=ARGS, dataset=dataset, encoder=encoder, decoder=decoder, transformer=transformer, data_generator=test_data_generator, exp_name=ARGS.exp_name, dset_fragment='test', fixed_thresh=fixed_thresh, setting=ARGS.setting)
        print('TEST')
        print(test_info)

