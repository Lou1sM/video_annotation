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
    exp_name = utils.get_datetime_stamp() if args.exp_name == "" else args.exp_name
    if args.enc_dec_hidden_init and (args.enc_size != args.dec_size):
        print("Not applying enc_dec_hidden_init because the encoder and decoder are different sizes")
        args.enc_dec_hidden_init = False

    if args.mini:
        train_table = val_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=args.enc_cnn)
        args.batch_size = 2
        train_file_path = val_file_path = test_file_path = '../data/rdf_video_captions/50d.6dp.h5'
    else:
        train_table = video_lookup_table_from_range(1,1201, cnn=args.enc_cnn)
        val_table = video_lookup_table_from_range(1201,1301, cnn=args.enc_cnn)
    
        train_file_path = os.path.join('../data/rdf_video_captions', 'train_{}.h5'.format(args.dataset))
        val_file_path = os.path.join('../data/rdf_video_captions', 'val_{}.h5'.format(args.dataset))
        test_file_path = os.path.join('../data/rdf_video_captions', 'test_{}.h5'.format(args.dataset))

    train_generator = load_data_lookup(train_file_path, video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
    val_generator = load_data_lookup(val_file_path, video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)

    print(args)
    
    if args.verbose:
        print("\nENCODER")
        print(encoder)
        print("\nDECODER")
        print(decoder)
        print("\nREGRESSOR")
        print(regressor)

    device = args.device

    if args.model == 'seq2seq':
        if args.reload_path:
            print('Reloading model from {}'.format(args.reload_path))
            saved_model = torch.load(args.reload_path)
            encoder = saved_model['encoder']
            decoder = saved_model['decoder']
            encoder_optimizer = saved_model['encoder_optimizer']
            decoder_optimizer = saved_model['decoder_optimizer']
        else: 
            encoder = models_train.EncoderRNN(args, device).to(device)
            decoder = models_train.DecoderRNN(args, device).to(device)
            encoder_optimizer = None
            decoder_optimizer = None
      
        models_train.train(args, encoder, decoder, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=device, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer)
       
        test_table = video_lookup_table_from_range(1301,1971, cnn=args.enc_cnn)
        train_generator = load_data_lookup(train_file_path, video_lookup_table=train_table, batch_size=1, shuffle=False)
        val_generator = load_data_lookup(val_file_path, video_lookup_table=val_table, batch_size=1, shuffle=False)
        test_generator = load_data_lookup(test_file_path, video_lookup_table=test_table, batch_size=1, shuffle=False)

        if args.chkpt:
            print("Reloading best network version for outputs")
            checkpoint = torch.load("../checkpoints/chkpt{}.pt".format(exp_name))
            encoder = checkpoint['encoder']
            decoder = checkpoint['decoder']
        else:
            print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
        
        print('\nComputing outputs on train set')
        train_preds = get_outputs_and_info(encoder, decoder, enc_zeroes=args.enc_zeroes, dec_zeroes=args.dec_zeroes, teacher_forcing=False, ind_size=args.ind_size, data_generator=train_generator, device=args.device, test_name='train_{}'.format(exp_name))
        print('\nComputing outputs on val set')
        val_preds = get_outputs_and_info(encoder, decoder, enc_zeroes=args.enc_zeroes, dec_zeroes=args.dec_zeroes, teacher_forcing=False, ind_size=args.ind_size, data_generator=val_generator, device=args.device, test_name='val_{}'.format(exp_name))
        print('\nComputing outputs on test set')
        test_preds = get_outputs_and_info(encoder, decoder, enc_zeroes=args.enc_zeroes, dec_zeroes=args.dec_zeroes, teacher_forcing=False, ind_size=args.ind_size, data_generator=test_generator, device=args.device, test_name='test_{}'.format(exp_name))
        
            
    elif args.model == 'reg':
        checkpoint = torch.load("../checkpoints/chkpt05-08_18:16:17.pt")
        print("\n begin checkpoint")
        print(checkpoint)
        print("\n end \n")
        encoder = checkpoint['encoder']
        regressor = models.NumIndRegressor(args,device).to(device)
        models.train_iters_reg(args, encoder, regressor, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=device)
    elif args.model == 'eos':
        checkpoint = torch.load("../checkpoints/chkpt05-08_18:16:17.pt")
        encoder = checkpoint['encoder']
        eos = models.NumIndEOS(args, device).to(device)
        models.train_iters_eos(args, encoder, eos, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=device)
    


if __name__=="__main__":
    args = options.load_arguments()
    main()
