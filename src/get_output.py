import math
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
from torchvision import models
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids, i3d_lookup_table_from_range
from search_threshes import find_best_thresh_from_probs, compute_scores_for_thresh
from sklearn.metrics import confusion_matrix

#torch.manual_seed(0)

def test_reg(encoder, regressor, train_generator, val_generator, test_generator, deci3d, device):
    encoder.eval()
    regressor.eval()

    y_gt_train = []
    y_pred_train = []

    y_gt_val = []
    y_pred_val = []
    y_pred_val_float = []

    y_gt_test = []
    y_pred_test = []

    y_gt_total = []
    y_pred_total = []
        
    for iter_, val_batch in enumerate(val_generator):
        input_ = val_batch[0].float().transpose(0,1).to(device)
        target_number = val_batch[2].float().to(device)
        i3d_vec = val_batch[5].float().to(device)

        encoder_hidden = encoder.initHidden().to(device)
        cnn_outputs = torch.zeros(encoder.num_frames, input_.shape[1], encoder.output_cnn_size+encoder.i3d_size, device=encoder.device)

        for i, inp in enumerate(input_):
            embedded = encoder.cnn_vector(inp)
            if encoder.i3d:
                embedded = torch.cat([embedded, i3d_vec], dim=1)
            cnn_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        encoder_outputs, encoder_hidden = encoder.rnn(cnn_outputs, encoder_hidden)
        #encoder_outputs, encoder_hidden = encoder(input_, i3d_vec, encoder_hidden)

        vid_vec = encoder_outputs.mean(dim=0)

        if deci3d:
            vid_vec = torch.cat([vid_vec, i3d_vec], dim=-1)
        print(vid_vec.shape)
        reg_pred = regressor(vid_vec)
        print('test', target_number, reg_pred)

        y_gt_val += target_number.tolist()
        y_pred_val_float += reg_pred.tolist()
        y_pred_val += [int(round(pred)) for pred in reg_pred.tolist()]
        y_gt_total += target_number.tolist()
        y_pred_total += [int(round(pred)) for pred in reg_pred.tolist()]
 
    accuracy_list_val = [y_gt_val[i] == y_pred_val[i] for i in range(len(y_gt_val))]
    accuracy_val = sum(accuracy_list_val)/len(accuracy_list_val)

    val_conf_mat = confusion_matrix(y_gt_val, y_pred_val)

    for iter_, train_batch in enumerate(train_generator):
        input_ = train_batch[0].float().transpose(0,1).to(device)
        target_number = train_batch[2].float().to(device)
        i3d_vec = train_batch[5].float().to(device)

        encoder_hidden = encoder.initHidden().to(device)
        for i, inp in enumerate(input_):
            embedded = encoder.cnn_vector(inp)
            if encoder.i3d:
                embedded = torch.cat([embedded, i3d_vec], dim=1)
            cnn_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        encoder_outputs, encoder_hidden = encoder.rnn(cnn_outputs, encoder_hidden)
        #encoder_outputs, encoder_hidden = encoder(input_, i3d_vec, encoder_hidden)

        vid_vec = encoder_outputs.mean(dim=0)
        print(encoder.rnn)
        print(vid_vec.shape)
        if deci3d:
            vid_vec = torch.cat([vid_vec, i3d_vec], dim=-1)
        print(vid_vec.shape)
        #encoder_outputs, encoder_hidden = encoder(input_, i3d_vec, encoder_hidden)

        vid_vec = encoder_outputs.mean(dim=0)

        if deci3d:
            vid_vec = torch.cat([vid_vec, i3d_vec], dim=-1)
        reg_pred = regressor(vid_vec)

        y_gt_train += target_number.tolist()
        y_pred_train += [int(round(pred)) for pred in reg_pred.tolist()]
        y_gt_total += target_number.tolist()
        y_pred_total += [int(round(pred)) for pred in reg_pred.tolist()]

    train_conf_mat = confusion_matrix(y_gt_train, y_pred_train)
    print(train_conf_mat)
 
    accuracy_list_train =  [y_gt_train[i] == y_pred_train[i] for i in range(len(y_gt_train))]
    accuracy_train = sum(accuracy_list_train)/len(accuracy_list_train)

    for iter_, test_batch in enumerate(test_generator):
        input_ = test_batch[0].float().transpose(0,1).to(device)
        target_number = test_batch[2].float().to(device)
        i3d_vec = test_batch[5].float().to(device)

        encoder_hidden = encoder.initHidden().to(device)
        for i, inp in enumerate(input_):
            embedded = encoder.cnn_vector(inp)
            if encoder.i3d:
                embedded = torch.cat([embedded, i3d_vec], dim=1)
            cnn_outputs[i] = embedded 

        # pass the output of the vgg layers through the GRU cell
        encoder_outputs, encoder_hidden = encoder.rnn(cnn_outputs, encoder_hidden)
        #encoder_outputs, encoder_hidden = encoder(input_, i3d_vec, encoder_hidden)

        vid_vec = encoder_outputs.mean(dim=0)

        if deci3d:
            vid_vec = torch.cat([vid_vec, i3d_vec], dim=-1)
        print(vid_vec.shape)
        #encoder_outputs, encoder_hidden = encoder(input_, i3d_vec, encoder_hidden)

        vid_vec = encoder_outputs.mean(dim=0)

        if deci3d:
            vid_vec = torch.cat([vid_vec, i3d_vec], dim=-1)
        reg_pred = regressor(vid_vec)

        y_gt_test += target_number.tolist()
        y_pred_test += [int(round(pred)) for pred in reg_pred.tolist()]
        y_gt_total += target_number.tolist()
        y_pred_total += [int(round(pred)) for pred in reg_pred.tolist()]

    test_conf_mat = confusion_matrix(y_gt_test, y_pred_test)
    print(test_conf_mat)
 
    accuracy_list_test = [y_gt_test[i] == y_pred_test[i] for i in range(len(y_gt_test))]
    accuracy_test = sum(accuracy_list_test)/len(accuracy_list_test)
 
    accuracy_list_total = [y_gt_total[i] == y_pred_total[i] for i in range(len(y_gt_total))]
    accuracy_total = sum(accuracy_list_total)/len(accuracy_list_total)
    total_conf_mat = confusion_matrix(y_gt_total, y_pred_total)

    return {'val_accuracy': accuracy_val, 'train_accuracy': accuracy_train, 'test_accuracy': accuracy_test, 'total_accuracy': accuracy_total}

def get_outputs(encoder, decoder, transformer, data_generator, gt_forcing, ind_size, setting='embeddings', deci3d=False, device='cuda'):
    encoder.eval()
    decoder.eval()
    encoder.batch_size = 1
    decoder.batch_size = 1
    decoder.i3d = deci3d
                
    criterion = nn.MSELoss()

    norms = []
    output = []
    positions = []
    total_l2_distances = []
    cos_sims = []
    eos_half_results = []
    eos_diff_results = []
    eos_diff_guesses = []
    eos_half_guesses = []
    eos_gts = []
    nums_of_inds = {}
    tup_sizes_by_pos = {}
    test_info = {}
    
    for iter_, dp in enumerate(data_generator):
        input_tensor = dp[0].float().transpose(0,1).to(device)
        target_tensor = dp[1].float().transpose(0,1).to(device)
        target_number = dp[2].float().to(device).int()
        video_id = dp[4].item()
        i3d = dp[5].float().to(device)
        
        if setting == "transformer":
            cnn = models.vgg19(pretrained=True).cuda()
            cnn_outputs = torch.zeros(8, input_tensor.shape[1], 4096, device=device)
            for i, inp in enumerate(input_tensor):
                x = cnn.features(inp)
                x = cnn.avgpool(x)
                x = x.view(x.size(0), -1)
                x = cnn.classifier[0](x)
                cnn_outputs[i] = x
            cnn_outputs = cnn_outputs.permute(1,0,2)
            target_tensor_perm = target_tensor.permute(1,0,2)

            growing_output = torch.zeros(1, 1, ind_size, device=device)
            next_transformer_preds = growing_output
            l=0
            while True:
                l+=1
                next_transformer_preds = transformer(cnn_outputs, next_transformer_preds)
                if l == target_number:
                    break
                next_transformer_preds = torch.cat([next_transformer_preds, growing_output], dim=1)
            #t_loss = criterion(next_transformer_preds, target_tensor_perm[:,:target_number].unsqueeze(0))
            l2_distances = torch.norm(next_transformer_preds.squeeze() - target_tensor_perm[:,:target_number].squeeze(), 2, dim=-1).tolist()
            output_norms = torch.norm(next_transformer_preds, 2, dim=-1)
            norm_criterion = nn.MSELoss()
            norm_loss = norm_criterion(torch.ones(1, target_number, device=device), output_norms)
            #eos_pred_list = list(next_transformer_preds[:,:,-1].squeeze().squeeze())
            eos_pred_tensor = next_transformer_preds[:,:,-1].squeeze()
            eos_pred_list = eos_pred_tensor.tolist()
            new_cos_sims = F.cosine_similarity(next_transformer_preds, target_tensor_perm[:, :target_number]).squeeze().tolist()
            cos_sims += new_cos_sims
            dp_output = next_transformer_preds.squeeze().tolist()
            gt_embeddings = target_tensor[:target_number].squeeze().tolist()
            new_norms = output_norms.squeeze().tolist()
            norms+=new_norms
            for i in range(target_number):
                try:
                    nums_of_inds[i] += 1
                except KeyError:
                    nums_of_inds[i] = 1
                new_norm = torch.norm(next_transformer_preds[0,i]).item()
                try:
                    old_num, old_denom = tup_sizes_by_pos[l]
                    tup_sizes_by_pos[i] = old_num+new_norm, old_denom+1
                except KeyError:
                    tup_sizes_by_pos[i] = new_norm, 1
                positions.append(i+1)

        elif setting in ["embeddings", "preds", 'eos']:
            encoder_hidden = encoder.initHidden()
            encoder_outputs, encoder_hidden = encoder(input_tensor, i3d, encoder_hidden)

            decoder_input = torch.zeros(1, 1, ind_size, device=decoder.device).to(device)
            if encoder.num_layers == decoder.num_layers and False:
                decoder_hidden = encoder_hidden
            else:
                decoder_hidden = decoder.initHidden()
            dp_output = []
            gt_embeddings = []
            l2_distances = []
            l_loss = 0
            eos_pred_list = []
            dec_out_list = []
            for l in range(target_number):
                decoder_output, decoder_hidden_new = decoder(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden, i3d=i3d)
                #try:
                #    eos_pred, _hidden  = decoder.eos_preds(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden, i3d=i3d) 
                #    eos_pred_list.append(eos_pred.item())
                #except Exception as e:
                #    print(e)
                #    pass
                decoder_output = F.normalize(decoder_output, p=2, dim=-1)
                dec_out_list.append(decoder_output)
                decoder_hidden = decoder_hidden_new
                dp_output.append(decoder_output.squeeze().detach().cpu().numpy().tolist())
                gt_embeddings.append(target_tensor[l].squeeze().detach().cpu().numpy().tolist())
                new_norm = torch.norm(decoder_output).item()
                norms.append(new_norm)

                #emb_pred = decoder_output[:,:,:-1]
                emb_pred = decoder_output
                #eos_pred = decoder_output[:,:,-1]
                #eos_pred_list.append(eos_pred.item())

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
                    #decoder_input = decoder_output
                    decoder_input = emb_pred
                #loss = criterion(decoder_output.squeeze(), target_tensor[l].squeeze())
                loss = criterion(emb_pred.squeeze(), target_tensor[l].squeeze())
                l_loss += loss
                #l2 = torch.norm(decoder_output.squeeze() - target_tensor[l].squeeze(), 2).item()
                l2 = torch.norm(emb_pred.squeeze() - target_tensor[l].squeeze(), 2).item()
                l2_distances.append(l2)
                #cos_sim = F.cosine_similarity(decoder_output.squeeze(), target_tensor[l].squeeze(),0).item()
                cos_sim = F.cosine_similarity(emb_pred.squeeze(), target_tensor[l].squeeze(),0).item()
                cos_sims.append(cos_sim)
                positions.append(l+1)
            #try:
            #    for l in range(target_number, 29):
            #        decoder_output, decoder_hidden_new = decoder(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden, i3d=i3d) 
            #        #emb_pred = decoder_output[:,:,:-1]
            #        #eos_pred = decoder_output[:,:,-1]
            #        eos_pred, decoder_hidden_new = decoder.eos_preds(input_=decoder_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs, hidden=decoder_hidden, i3d=i3d) 
            #        eos_pred_list.append(eos_pred.item())
            #        decoder_hidden = decoder_hidden_new
            #        #decoder_input = decoder_output
            #        #decoder_input = emb_pred
            #except Exception as e:
            #    print(e)
            #    pass
        #try:
        #    # Take first element that's greater that 0.5
        #    #eos_guess = [i>0.5 for i in eos_pred_list].index(True)
        #    eos_half_guess_list = [i>0 for i in eos_pred_list]
        #    eos_half_guess = eos_half_guess_list.index(True) + 1 if True in eos_half_guess_list else len(eos_half_guess_list)
        #    eos_cumulative_probs = [0] + [1 / (1 + math.exp(-x)) for x in eos_pred_list]
        #    diff_cum_probs = [eos_cumulative_probs[i+1] - eos_cumulative_probs[i] for i in range(len(eos_cumulative_probs)-1)]
        #    eos_diff_guess = np.argmax([eos_cumulative_probs[i+1] - eos_cumulative_probs[i] for i in range(len(eos_cumulative_probs)-1)]) + 1
        #    eos_diff_guess = np.argmax(diff_cum_probs) + 1
        #    eos_diff_guesses.append(eos_diff_guess)
        #    eos_half_guesses.append(eos_half_guess)
        #    eos_gt = target_number.item()
        #    eos_gts.append(eos_gt)
        #    eos_half_result = (eos_half_guess == eos_gt)
        #    eos_half_results.append(eos_half_result)
        #    eos_diff_result = (eos_diff_guess == eos_gt)
        #    eos_diff_results.append(eos_diff_result)
        #    if iter_%10 == 0:
        #        #print(eos_pred_list, eos_guess, target_number.item()-1, eos_result)
        #        print(eos_pred_list)
        #        #print(eos_cumulative_probs)
        #        #print(diff_cum_probs)
        #        print( 'gt', eos_gt, 'half', eos_half_guess, eos_half_result, 'diff', eos_diff_guess, eos_diff_result)
        #        #print( 'gt', eos_gt, 'diff', eos_diff_guess, eos_diff_result)
        #except (ValueError, AttributeError) as e:
        #    eos_half_guess = eos_diff_guess = eos_target = eos_half_result = eos_diff_result = -1
        #    pass
        assert len(dp_output) == len(gt_embeddings), "Wrong number of embeddings in output: {} being output but {} in ground truth".format(len(dp_output), len(gt_embeddings))
        #output.append({'videoId':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'l2_distances': l2_distances, 'avg_l2_distance': sum(l2_distances)/len(l2_distances), 'eos_half_guess': str(eos_half_guess), 'eos_diff_guess': str(eos_diff_guess),'eos_gt': str(target_number.item()), 'eos_half_result': str(eos_half_result), 'eos_half_result': str(eos_half_result)})
        output.append({'videoId':str(video_id), 'embeddings':dp_output, 'gt_embeddings': gt_embeddings, 'l2_distances': l2_distances, 'avg_l2_distance': sum(l2_distances)/len(l2_distances)}) 
  
    #print(len(eos_gts), len(eos_half_guesses), len(eos_diff_guesses))

    #plt.scatter(eos_gts, eos_half_guesses)
    #plt.savefig('eos_half_scatter.png')
    #plt.scatter(eos_gts, eos_diff_guesses)
    #plt.savefig('eos_diff_scatter.png')
    assert sum(list(nums_of_inds.values())) == len(norms), "Mismatch in the lengths of nums_of_inds and norms, {} vs {}".format(len(nums_of_inds.values()), len(norms))
    avg_l2_distance = round(sum(l2_distances)/len(l2_distances),4)
    avg_cos_sim = round(sum(cos_sims)/len(cos_sims),4)
    avg_norm = round( (sum([t[0] for t in tup_sizes_by_pos.values()])/sum(nums_of_inds.values())),4)
    #eos_diff_accuracy = sum(eos_diff_results)/(len(eos_diff_results)+1e-5)
    #eos_half_accuracy = sum(eos_half_results)/(len(eos_half_results)+1e-5)
    test_info['l2_distance'] = avg_l2_distance
    test_info['cos_similarity'] = avg_cos_sim
    test_info['avg_norm'] = avg_norm
    #test_info['eos_diff_accuracy'] = eos_diff_accuracy
    #test_info['eos_half_accuracy'] = eos_half_accuracy
    for k,v in nums_of_inds.items():
        print(k,v)
    sizes_by_pos = {k: v[0]/v[1] for k,v in tup_sizes_by_pos.items()}
    for k,v in sizes_by_pos.items():
        print(k,v)
    return output, norms, positions, sizes_by_pos, test_info


def write_outputs_get_info(encoder, decoder, transformer, ARGS, data_generator, gt_forcing, exp_name, dset_fragment, fixed_thresh=None, setting='embeddings', deci3d=False):

    deci3d = ARGS.i3d and ARGS.i3d_after
    outputs, norms, positions, sizes_by_pos, test_info  = get_outputs(encoder, decoder, transformer, gt_forcing=gt_forcing ,data_generator=data_generator, ind_size=ARGS.ind_size, device=ARGS.device, setting=setting, deci3d=deci3d)

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
   
    metric_data, total_metric_data, positive_probs, negative_probs = find_best_thresh_from_probs(exp_name, dset_fragment, ind_size=ARGS.ind_size)
    test_info.update(metric_data)
    legit_thresh = fixed_thresh if dset_fragment == 'test' else metric_data['thresh'] 
    print(fixed_thresh, metric_data['thresh'])
    test_info['legit_f1'] = compute_scores_for_thresh(positive_probs, negative_probs, legit_thresh)[6]
    
    plt.xlim(0,len(total_metric_data['thresh']))
    plt.ylim(0.0, max(total_metric_data['f1']) + 0.1)
    plt.plot(total_metric_data['f1'], label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title("{}: F1 scores by threshold".format(exp_name))
    plt.savefig('../experiments/{}/{}-{}_F1_scores_by_threshold.png'.format(exp_name, exp_name, dset_fragment))
    plt.clf()

    print(test_info)
    return sizes_by_pos, test_info
 
if __name__=="__main__":

    device='cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_forcing", "-gtf",  action="store_true")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--enc_zeroes", action="store_true")
    parser.add_argument("--dec_zeroes", action="store_true")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--ind_size", type=int, default=10)
    parser.add_argument("--i3d", action="store_true")
    parser.add_argument("--i3d_after", action="store_true")
    parser.add_argument("--setting", type=str, default='embeddings')
    parser.add_argument("--checkpoint_to_test", type=str)


    ARGS = parser.parse_args() 
    ARGS.device = "cuda" if torch.cuda.is_available() else "cpu"
    #checkpoint_path = '/data2/louis/checkpoints/{}.pt'.format(ARGS.exp_name)
    #checkpoint_path = '../jade_checkpoints/{}.pt'.format(ARGS.exp_name)
    checkpoint_path = ARGS.checkpoint_to_test
        
    gtf = ARGS.gt_forcing
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

    train_norms = []
    val_norms = []
    test_norms = []

    exp_dir = '../experiments/{}'.format(ARGS.exp_name)
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    print('getting outputs and info for val set')
    val_lookup_table = video_lookup_table_from_range(1201, 1301, cnn="vgg")
    i3d_val_lookup_table = i3d_lookup_table_from_range(1201, 1301) if ARGS.i3d else None
    val_data_generator = load_data_lookup('../data/rdf_video_captions/{}d-val.h5'.format(ARGS.ind_size), video_lookup_table=val_lookup_table, i3d_lookup_table=i3d_val_lookup_table, batch_size=1, shuffle=False)
    _, val_info = write_outputs_get_info(ARGS=ARGS, encoder=encoder, decoder=decoder, transformer=transformer, data_generator=val_data_generator, gt_forcing=gtf, exp_name=ARGS.exp_name, dset_fragment='val', setting=ARGS.setting)


    if not ARGS.quick:
        print('getting outputs and info for train set')
        train_lookup_table = video_lookup_table_from_range(1, 1201, cnn="vgg")
        i3d_train_lookup_table = i3d_lookup_table_from_range(1, 1201) if ARGS.i3d else None
        train_data_generator = load_data_lookup('../data/rdf_video_captions/{}d-train.h5'.format(ARGS.ind_size), video_lookup_table=train_lookup_table, i3d_lookup_table=i3d_train_lookup_table, batch_size=1, shuffle=False)
        _, train_info = write_outputs_get_info(ARGS=ARGS, encoder=encoder, decoder=decoder, transformer=transformer, data_generator=train_data_generator, gt_forcing=gtf, exp_name=ARGS.exp_name, dset_fragment='train', setting=ARGS.setting)


        print('getting outputs and info for test set')
        fixed_thresh = ((train_info['thresh']*1200)+(val_info['thresh']*100))/1300
        print('outer_fixed_thresh', fixed_thresh)
        test_lookup_table = video_lookup_table_from_range(1301, 1971, cnn="vgg")
        i3d_test_lookup_table = i3d_lookup_table_from_range(1301, 1971) if ARGS.i3d else None
        test_data_generator = load_data_lookup('../data/rdf_video_captions/{}d-test.h5'.format(ARGS.ind_size), video_lookup_table=test_lookup_table, i3d_lookup_table=i3d_test_lookup_table, batch_size=1, shuffle=False)
        _, test_info = write_outputs_get_info(ARGS=ARGS, encoder=encoder, decoder=decoder, transformer=transformer, data_generator=test_data_generator, gt_forcing=gtf, exp_name=ARGS.exp_name, dset_fragment='test', fixed_thresh=fixed_thresh, setting=ARGS.setting)
        print('TEST')
        print(test_info)

