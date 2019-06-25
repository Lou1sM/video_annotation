import csv
import sys
import os
import json
import utils
import options
import models
import train
import torch
from data_loader import load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids
from get_output import write_outputs_get_info



def run_experiment(exp_name, ARGS, train_table, val_table, test_table):
    """Cant' just pass generators as need to re-init with batch_size=1 when testing.""" 
    
    dataset = '{}d'.format(ARGS.ind_size)
    
    if ARGS.mini:
        ARGS.batch_size = min(2, ARGS.batch_size)
        ARGS.enc_size = ARGS.dec_size = 50
        ARGS.enc_layers = ARGS.dec_layers = 1
        #ARGS.ind_size = 10
        ARGS.no_chkpt = True
        if ARGS.max_epochs == 1000:
            ARGS.max_epochs = 1
        train_file_path = val_file_path = test_file_path = '../data/rdf_video_captions/{}d-6dp.h5'.format(ARGS.ind_size)
        assert os.path.isfile(train_file_path)
        print('Using dataset: {}'.format(train_file_path))
    else:
        train_file_path = os.path.join('../data/rdf_video_captions', '{}-train.h5'.format(dataset))
        val_file_path = os.path.join('../data/rdf_video_captions', '{}-val.h5'.format(dataset))
        test_file_path = os.path.join('../data/rdf_video_captions', '{}-test.h5'.format(dataset))
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
            encoder.batch_size = ARGS.batch_size
            decoder.batch_size = ARGS.batch_size
            encoder_optimizer = saved_model['encoder_optimizer']
            decoder_optimizer = saved_model['decoder_optimizer']
        else: 
            encoder = models.EncoderRNN(ARGS, ARGS.device).to(ARGS.device)
            #decoder = models.DecoderRNN(ARGS, ARGS.device).to(ARGS.device)
            decoder = models.DecoderRNN_openattn(ARGS).to(ARGS.device)
            encoder_optimizer = None
            decoder_optimizer = None
      
        print('\nTraining the model')
        train_info, _ = train.train(ARGS, encoder, decoder, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=ARGS.device, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer)
       
        train_generator = load_data_lookup(train_file_path, video_lookup_table=train_table, batch_size=1, shuffle=False)
        val_generator = load_data_lookup(val_file_path, video_lookup_table=val_table, batch_size=1, shuffle=False)
        test_generator = load_data_lookup(test_file_path, video_lookup_table=test_table, batch_size=1, shuffle=False)

      
        if ARGS.no_chkpt:
            print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
        else:
            print("Reloading best network version for outputs")
            checkpoint = torch.load("../checkpoints/{}.pt".format(exp_name))
            encoder = checkpoint['encoder']
            decoder = checkpoint['decoder']

        print('\nComputing outputs on val set')
        val_sizes_by_pos, val_output_info = write_outputs_get_info(encoder, decoder, ARGS, gt_forcing=False, data_generator=val_generator, exp_name=exp_name, dset_fragment='val')
        val_sizes_by_pos['dset_fragment'] = 'val'
        val_output_info['dset_fragment'] = 'val'
        print('\nComputing outputs on train set')
        train_sizes_by_pos, train_output_info = write_outputs_get_info(encoder, decoder, ARGS, gt_forcing=False, data_generator=train_generator, exp_name=exp_name, dset_fragment='train')
        train_sizes_by_pos['dset_fragment'] = 'train'
        train_output_info['dset_fragment'] = 'train'
        fixed_thresh = ((train_output_info['thresh']*1200)+(val_output_info['thresh']*100))/1300
        print('\nComputing outputs on test set')
        test_sizes_by_pos, test_output_info = write_outputs_get_info(encoder, decoder, ARGS, gt_forcing=False, data_generator=test_generator, exp_name=exp_name, dset_fragment='test', fixed_thresh=fixed_thresh)
        test_sizes_by_pos['dset_fragment'] = 'test'
        test_output_info['dset_fragment'] = 'test'
  
        pos_norms_csv_filename = '../experiments/{}/{}_avg_norms_position.csv'.format(exp_name, exp_name)
        with open(pos_norms_csv_filename, 'w') as csv_file:
            w = csv.DictWriter(csv_file, fieldnames=['dset_fragment']+list(range(len(train_sizes_by_pos))))
            w.writerow(train_sizes_by_pos)
            w.writerow(val_sizes_by_pos)
            w.writerow(test_sizes_by_pos)
         
        summary_filename = '../experiments/{}/{}_summary.txt'.format(exp_name, exp_name) 
        with open(summary_filename, 'w') as summary_file:
            summary_file.write('Experiment name: {}\n'.format(exp_name))
            summary_file.write('\tTrain\tVal\tTest\n')
            #for k in train_output_info:
            for k in ['dset_fragment', 'l2_distance', 'cos_similarity', 'avg_norm', 'thresh', 'legit_f1', 'eos_accuracy', 'avg_pos_prob', 'avg_neg_prob']: 
                summary_file.write(k+'\t'+str(train_output_info[k])+'\t'+str(val_output_info[k])+'\t'+str(test_output_info[k])+'\n')
            summary_file.write('\nParameters:\n')
            for key in options.IMPORTANT_PARAMS:
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
        if ARGS.reload_path:
            encoder = torch.load(ARGS.reload_path)['encoder']
        else:
            encoder = models.EncoderRNN(ARGS, ARGS.device).to(ARGS.device)
        eos = models.NumIndEOS(ARGS).to(ARGS.device)
        train.train_iters_eos(ARGS, encoder, eos, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=ARGS.device)
 
    accuracy = 0
    return accuracy, test_output_info


def get_user_yesno_answer(question):
    answer = input(question+'(y/n)')
    if answer == 'y':
        return True
    elif answer == 'n':
        return False
    else:
        print("Please answer 'y' or 'n'")
        return(get_user_yesno_answer(question))
    

def main():
    #dummy_output = 10
    if ARGS.mini:
        ARGS.exp_name = 'try'
    
    exp_name = utils.get_datetime_stamp() if ARGS.exp_name == "" else ARGS.exp_name
    if ARGS.exp_name == 'try':
        pass
    elif os.path.isdir('../experiments/{}'.format(exp_name)):
        try:
            overwrite = get_user_yesno_answer('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name))
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
        #train_table = val_table = test_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=ARGS.enc_cnn)
        train_table = val_table = test_table = video_lookup_table_from_range(1,7, cnn=ARGS.enc_cnn)
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
