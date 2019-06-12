import json
import utils
import options
import models
import models_working
import torch
from data_loader import load_data_lookup, video_lookup_table_from_range, video_lookup_table_from_ids
from get_output import get_output_gen


def main():
    #dummy_output = 10
    exp_name = utils.get_datetime_stamp() if args.exp_name == "" else args.exp_name
    if args.enc_dec_hidden_init and (args.enc_size != args.dec_size):
        print("Not applying enc_dec_hidden_init because the encoder and decoder are different sizes")
        args.enc_dec_hidden_init = False

    if args.mini:
        #vid_table = video_lookup_table_from_range(1,4, cnn=args.enc_cnn)
        train_table = val_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=args.enc_cnn)
        args.batch_size = 2
        args.h5_train_file_path = '../data/rdf_video_captions/50d.6dp.h5'
        args.h5_val_file_path = '../data/rdf_video_captions/50d.6dp.h5'

        #h5_train_generator = load_data_lookup('../data/rdf_video_captions/50d.6dp.h5', video_lookup_table=vid_table, batch_size=args.batch_size, shuffle=args.shuffle)
        #h5_val_generator = load_data_lookup('../data/rdf_video_captions/50d.6dp.h5', video_lookup_table=vid_table, batch_size=args.batch_size, shuffle=args.shuffle)
        #h5_train_generator = load_data_lookup('/home/eleonora/video_annotation/data/rdf_video_captions/50d_overfitting.h5', video_lookup_table=vid_table, batch_size=args.batch_size, shuffle=args.shuffle)
        #h5_val_generator = load_data_lookup('/home/eleonora/video_annotation/data/rdf_video_captions/50d_overfitting.h5', video_lookup_table=vid_table, batch_size=args.batch_size, shuffle=False)
    else:
        train_table = video_lookup_table_from_range(1,1201, cnn=args.enc_cnn)
        val_table = video_lookup_table_from_range(1201,1301, cnn=args.enc_cnn)
        #vid_table = video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=args.enc_cnn)
        #train_table = val_table = vid_table
    h5_train_generator = load_data_lookup(args.h5_train_file_path, video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
    h5_val_generator = load_data_lookup(args.h5_val_file_path, video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)

    print(args)
    
    if args.verbose:
        print("\nENCODER")
        print(encoder)
        print("\nDECODER")
        print(decoder)
        print("\nREGRESSOR")
        print(regressor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            encoder = models_working.EncoderRNN(args, device).to(device)
            decoder = models_working.DecoderRNN(args, device).to(device)
            encoder_optimizer = None
            decoder_optimizer = None
        models_working.train(args, encoder, decoder, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer)
        h5_train_generator = load_data_lookup(args.h5_train_file_path, video_lookup_table=train_table, batch_size=1, shuffle=False)
        h5_val_generator = load_data_lookup(args.h5_val_file_path, video_lookup_table=val_table, batch_size=1, shuffle=False)
        checkpoint = torch.load("../checkpoints/chkpt{}.pt".format(exp_name))
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        print('\nGetting train outputs')
        train_preds = get_output_gen(encoder, decoder, ind_size=args.ind_size, data_generator=h5_train_generator)
        print('\nGetting val outputs')
        val_preds = get_output_gen(encoder, decoder, ind_size=args.ind_size, data_generator=h5_val_generator)
        
        with open('../data/test_outputs/train_preds_{}.txt'.format(exp_name), 'w') as train_out_file:
            json.dump(train_preds, train_out_file)
        with open('../data/test_outputs/val_preds_{}.txt'.format(exp_name), 'w') as val_out_file:
            json.dump(val_preds, val_out_file)
            
    elif args.model == 'reg':
        checkpoint = torch.load("../checkpoints/chkpt05-08_18:16:17.pt")
        print("\n begin checkpoint")
        print(checkpoint)
        print("\n end \n")
        encoder = checkpoint['encoder']
        regressor = models.NumIndRegressor(args,device).to(device)
        models.train_iters_reg(args, encoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    elif args.model == 'eos':
        checkpoint = torch.load("../checkpoints/chkpt05-08_18:16:17.pt")
        encoder = checkpoint['encoder']
        eos = models.NumIndEOS(args, device).to(device)
        models.train_iters_eos(args, encoder, eos, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    


if __name__=="__main__":
    args = options.load_arguments()
    main()
