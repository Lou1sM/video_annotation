import utils
import options
import models
import torch
from data_loader import load_data_lookup, video_lookup_table_from_range


def main():
    #dummy_output = 10
    exp_name = utils.get_datetime_stamp()
    print(args)
    if args.enc_dec_hidden_init and (args.enc_size != args.dec_size):
        print("Not applying enc_dec_hidden_init because the encoder and decoder are different sizes")
        args.enc_dec_hidden_init = False

    if args.mini:
        train_table = video_lookup_table_from_range(1,4)
        val_table = video_lookup_table_from_range(1,4)
        h5_train_generator = load_data_lookup('../data/50d_overfitting.h5', video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
        h5_val_generator = load_data_lookup('../data/50d_overfitting.h5', video_lookup_table=val_table, batch_size=args.batch_size, shuffle=False)
    else:
        train_table = video_lookup_table_from_range(1,1201)
        val_table = video_lookup_table_from_range(1201,1301)
        h5_train_generator = load_data_lookup(args.h5_train_file_path, video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
        h5_val_generator = load_data_lookup(args.h5_val_file_path, video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)

    
    if args.verbose:
        print("\nENCODER")
        print(encoder)
        print("\nDECODER")
        print(decoder)
        print("\nREGRESSOR")
        print(regressor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'seq2seq':
        if args.reload_path:
            print('Reloading model from {}'.format(args.reload_path))
            saved_model = torch.load(args.reload_path)
            encoder = saved_model['encoder']
            decoder = saved_model['decoder']
            encoder_optimizer = saved_model['encoder_optimizer']
            decoder_optimizer = saved_model['decoder_optimizer']
        else: 
            encoder = models.EncoderRNN(args, device).to(device)
            decoder = models.DecoderRNN(args, device).to(device)
            encoder_optimizer = None
            decoder_optimizer = None
        models.train(args, encoder, decoder, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer)
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
