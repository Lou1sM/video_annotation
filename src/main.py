import utils
import options
import models
import torch
from data_loader import load_data_lookup, video_lookup_table_from_range


def main():
    #dummy_output = 10
    exp_name = utils.get_datetime_stamp()
    print(args)
    if args.mini:
        train_table = video_lookup_table_from_range(1,11)
        val_table = video_lookup_table_from_range(1201,1211)
        h5_train_generator = load_data_lookup('../data/mini/train_data.h5', video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
        h5_val_generator = load_data_lookup('../data/mini/val_data.h5', video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)
    else:
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
    encoder = models.EncoderRNN(args, device).to(device)


    if args.model == 'seq2seq':
        decoder = models.DecoderRNN(args, device).to(device)
        models.train_iters_seq2seq(args, encoder, decoder, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    elif args.model == 'reg':
        regressor = models.NumIndRegressor(args,device).to(device)
        models.train_iters_reg(args, encoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    elif args.model == 'eos':
        eos = models.NumIndEOS(args, device).to(device)
        models.train_iters_eos(args, encoder, eos, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    


if __name__=="__main__":
    args = options.load_arguments()
    main()
