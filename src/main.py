import os
import json
import utils
import options
import models_train
import torch
from data_loader import load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids
from get_output import get_outputs_and_info


def main():
    #dummy_output = 10
    exp_name = utils.get_datetime_stamp() if ARGS.exp_name == "" else ARGS.exp_name
    if ARGS.enc_dec_hidden_init and (ARGS.enc_size != ARGS.dec_size):
        print("Not applying enc_dec_hidden_init because the encoder and decoder are different sizes")
        ARGS.enc_dec_hidden_init = False

    dataset = '{}d-{}'.format(ARGS.ind_size, ARGS.rrn_init)
    if ARGS.mini:
        train_table = val_table = test_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=ARGS.enc_cnn)
        ARGS.batch_size = 2
        ARGS.enc_size = ARGS.dec_size = 50
        ARGS.enc_layers = ARGS.dec_layers = 1
        ARGS.ind_size = 50
        train_file_path = val_file_path = test_file_path = '../data/rdf_video_captions/50d.6dp.h5'
    else:
        print('\nLoading lookup tables\n')
        train_table = video_lookup_table_from_range(1,1201, cnn=ARGS.enc_cnn)
        val_table = video_lookup_table_from_range(1201,1301, cnn=ARGS.enc_cnn)
        test_table = video_lookup_table_from_range(1301,1971, cnn=ARGS.enc_cnn)
    
        train_file_path = os.path.join('../data/rdf_video_captions', 'train_{}.h5'.format(dataset))
        val_file_path = os.path.join('../data/rdf_video_captions', 'val_{}.h5'.format(dataset))
        test_file_path = os.path.join('../data/rdf_video_captions', 'test_{}.h5'.format(dataset))

    train_generator = load_data_lookup(train_file_path, video_lookup_table=train_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)
    val_generator = load_data_lookup(val_file_path, video_lookup_table=val_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)

    print(ARGS)
    
    if ARGS.verbose:
        print("\nENCODER")
        print(encoder)
        print("\nDECODER")
        print(decoder)
        print("\nREGRESSOR")
        print(regressor)

    device = ARGS.device

    if ARGS.model == 'seq2seq':
        if ARGS.reload_path:
            print('Reloading model from {}'.format(ARGS.reload_path))
            saved_model = torch.load(ARGS.reload_path)
            encoder = saved_model['encoder']
            decoder = saved_model['decoder']
            encoder_optimizer = saved_model['encoder_optimizer']
            decoder_optimizer = saved_model['decoder_optimizer']
        else: 
            encoder = models_train.EncoderRNN(ARGS, device).to(device)
            decoder = models_train.DecoderRNN(ARGS, device).to(device)
            encoder_optimizer = None
            decoder_optimizer = None
      
        print('\nTraining the model')
        models_train.train(ARGS, encoder, decoder, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=device, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer)
       
        train_generator = load_data_lookup(train_file_path, video_lookup_table=train_table, batch_size=1, shuffle=False)
        val_generator = load_data_lookup(val_file_path, video_lookup_table=val_table, batch_size=1, shuffle=False)
        test_generator = load_data_lookup(test_file_path, video_lookup_table=test_table, batch_size=1, shuffle=False)

        if ARGS.reload_path:
            print("Reloading best network version for outputs")
            #checkpoint = torch.load("../checkpoints/chkpt{}.pt".format(exp_name))
            checkpoint = torch.load(ARGS.reload_path)
            encoder = checkpoint['encoder']
            decoder = checkpoint['decoder']
        else:
            print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
        
        print('\nComputing outputs on train set')
        train_preds = get_outputs_and_info(encoder, decoder, enc_zeroes=ARGS.enc_zeroes, dec_zeroes=ARGS.dec_zeroes, teacher_forcing=False, ind_size=ARGS.ind_size, data_generator=train_generator, device=ARGS.device, test_name='{}-train_{}'.format(dataset, exp_name))
        print('\nComputing outputs on val set')
        val_preds = get_outputs_and_info(encoder, decoder, enc_zeroes=ARGS.enc_zeroes, dec_zeroes=ARGS.dec_zeroes, teacher_forcing=False, ind_size=ARGS.ind_size, data_generator=val_generator, device=ARGS.device, test_name='{}-val_{}'.format(dataset, exp_name))
        print('\nComputing outputs on test set')
        test_preds = get_outputs_and_info(encoder, decoder, enc_zeroes=ARGS.enc_zeroes, dec_zeroes=ARGS.dec_zeroes, teacher_forcing=False, ind_size=ARGS.ind_size, data_generator=test_generator, device=ARGS.device, test_name='{}-test_{}'.format(dataset, exp_name))
        
            
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
    


if __name__=="__main__":
    ARGS = options.load_arguments()
    main()
