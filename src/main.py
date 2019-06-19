import csv
import sys
import os
import json
import utils
import options
import models_train
import torch
from data_loader import load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids
from get_output import write_outputs_get_info


def run_experiment(exp_name, ARGS, train_table, val_table, test_table):
    """Cant' just pass generators as need to re-init with batch_size=1 when testing.""" 
    
    dataset = '{}d-{}'.format(ARGS.ind_size, ARGS.rrn_init)
    
    if ARGS.mini:
        ARGS.batch_size = 2
        ARGS.enc_size = ARGS.dec_size = 50
        ARGS.enc_layers = ARGS.dec_layers = 1
        ARGS.ind_size = 50
        train_file_path = val_file_path = test_file_path = '../data/rdf_video_captions/50d.6dp.h5'
        print('Using dataset: ../data/rdf_video_captions/50d.6dp')
    else:
        train_file_path = os.path.join('../data/rdf_video_captions', 'train_{}.h5'.format(dataset))
        val_file_path = os.path.join('../data/rdf_video_captions', 'val_{}.h5'.format(dataset))
        test_file_path = os.path.join('../data/rdf_video_captions', 'test_{}.h5'.format(dataset))
        print('Using dataset: {}'.format(dataset))
   


    train_generator = load_data_lookup(train_file_path, video_lookup_table=train_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)
    val_generator = load_data_lookup(val_file_path, video_lookup_table=val_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)

    print(ARGS)
   
    if ARGS.model == 'seq2seq':
        if ARGS.reload_path:
            print('Reloading model from {}'.format(ARGS.reload_path))
            saved_model = torch.load(ARGS.reload_path)
            encoder = saved_model['encoder']
            decoder = saved_model['decoder']
            encoder_optimizer = saved_model['encoder_optimizer']
            decoder_optimizer = saved_model['decoder_optimizer']
        else: 
            encoder = models_train.EncoderRNN(ARGS, ARGS.device).to(ARGS.device)
            decoder = models_train.DecoderRNN(ARGS, ARGS.device).to(ARGS.device)
            encoder_optimizer = None
            decoder_optimizer = None
      
        print('\nTraining the model')
        train_info, _ = models_train.train(ARGS, encoder, decoder, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=ARGS.device, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer)
       
        train_generator = load_data_lookup(train_file_path, video_lookup_table=train_table, batch_size=1, shuffle=False)
        val_generator = load_data_lookup(val_file_path, video_lookup_table=val_table, batch_size=1, shuffle=False)
        test_generator = load_data_lookup(test_file_path, video_lookup_table=test_table, batch_size=1, shuffle=False)

      
        if ARGS.chkpt:
            print("Reloading best network version for outputs")
            checkpoint = torch.load("../checkpoints/{}.pt".format(exp_name))
            encoder = checkpoint['encoder']
            decoder = checkpoint['decoder']
        else:
            print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
        
        print('\nComputing outputs on train set')
        train_sizes_by_pos, train_output_info = write_outputs_get_info(encoder, decoder, ARGS, gt_forcing=False, data_generator=train_generator, exp_name=exp_name, test_name='{}-train_{}'.format(dataset, exp_name))
        train_sizes_by_pos['dset'] = 'train'
        train_output_info['dataset'] = 'train'
        print('\nComputing outputs on val set')
        val_sizes_by_pos, val_output_info = write_outputs_get_info(encoder, decoder, ARGS, gt_forcing=False, data_generator=val_generator, exp_name=exp_name, test_name='{}-val_{}'.format(dataset, exp_name))
        val_sizes_by_pos['dset'] = 'val'
        val_output_info['dataset'] = 'val'
        print('\nComputing outputs on test set')
        test_sizes_by_pos, test_output_info = write_outputs_get_info(encoder, decoder, ARGS, gt_forcing=False, data_generator=test_generator, exp_name=exp_name, test_name='{}-test_{}'.format(dataset, exp_name))
        test_sizes_by_pos['dset'] = 'test'
        test_output_info['dataset'] = 'test'
  
        pos_norms_csv_filename = '../experiments/{}/{}_avg_norms_position.csv'.format(exp_name, exp_name)
        with open(pos_norms_csv_filename, 'w') as csv_file:
            w = csv.DictWriter(csv_file, fieldnames=['dset']+list(range(len(train_sizes_by_pos))))
            w.writerow(train_sizes_by_pos)
            w.writerow(val_sizes_by_pos)
            w.writerow(test_sizes_by_pos)
         
        summary_filename = '../experiments/{}/{}_summary.txt'.format(exp_name, exp_name) 
        with open(summary_filename, 'w') as summary_file:
            summary_file.write('Experiment name: {}\n'.format(exp_name))
            summary_file.write('\tTrain\tVal\tTest\n')
            #for k in train_output_info:
            for k in ['dataset', 'l2_distance', 'cos_similarity', 'avg_emb_norm']: 
                summary_file.write(k+'\t'+str(train_output_info[k])+'\t'+str(val_output_info[k])+'\t'+str(test_output_info[k])+'\n')
            summary_file.write('\nParameters:\n')
            for key in sorted(vars(ARGS).keys()):
                summary_file.write(str(key) + ": " + str(vars(ARGS)[key]) + "\n")

            
    elif ARGS.model == 'reg':
        checkpoint = torch.load("../checkpoints/chkpt05-08_18:16:17.pt")
        print("\n begin checkpoint")
        print(checkpoint)
        print("\n end \n")
        encoder = checkpoint['encoder']
        regressor = models.NumIndRegressor(ARGS,device).to(device)
        models.train_iters_reg(ARGS, encoder, regressor, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=device)
    elif ARGS.model == 'eos':
        checkpoint = torch.load("../checkpoints/chkpt05-08_18:16:17.pt")
        encoder = checkpoint['encoder']
        eos = models.NumIndEOS(ARGS, device).to(device)
        models.train_iters_eos(ARGS, encoder, eos, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=device)
 
    accuracy = 0
    return accuracy, test_output_info


def main():
    #dummy_output = 10
    exp_name = utils.get_datetime_stamp() if ARGS.exp_name == "" else ARGS.exp_name
    if os.path.isdir('../experiments/{}'.format(exp_name)):
        try:
            overwrite = input('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name))
        except OSError:
            overwrite = ARGS.overwrite
        if not overwrite:
            print('Please rerun command with a different experiment name')
            sys.exit()
    else: 
        os.mkdir('../experiments/{}'.format(exp_name))

    if ARGS.enc_dec_hidden_init and (ARGS.enc_size != ARGS.dec_size):
        print("Not applying enc_dec_hidden_init because the encoder and decoder are different sizes")
        ARGS.enc_dec_hidden_init = False

    if ARGS.mini:
        train_table = val_table = test_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=ARGS.enc_cnn)
    else:
        print('\nLoading lookup tables\n')
        train_table = video_lookup_table_from_range(1,1201, cnn=ARGS.enc_cnn)
        val_table = video_lookup_table_from_range(1201,1301, cnn=ARGS.enc_cnn)
        test_table = video_lookup_table_from_range(1301,1971, cnn=ARGS.enc_cnn)
    
    run_experiment( exp_name, 
                    ARGS,
                    train_table=train_table,
                    val_table=val_table,
                    test_table=test_table)

if __name__=="__main__":
    ARGS = options.load_arguments()
    main()
