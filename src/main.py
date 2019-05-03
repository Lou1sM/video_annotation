import utils
import options
import seq2seq
import torch
from data_loader import load_data


def main():
    #dummy_output = 10
    exp_name = utils.get_datetime_stamp()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = seq2seq.EncoderRNN(args, device).to(device)
    decoder = seq2seq.DecoderRNN(args, device).to(device)
    regressor = seq2seq.NumIndRegressor(args,device).to(device)

    h5_train_generator = load_data(args.h5_file_path, args.batch_size, shuffle=args.shuffle)
    h5_val_generator = load_data(args.h5_file_path, args.batch_size, shuffle=args.shuffle)
    
    if args.verbose:
        print("\nENCODER")
        print(encoder)
        print("\nDECODER")
        print(decoder)
        print("\nREGRESSOR")
        print(regressor)

    seq2seq.trainIters(args, encoder, decoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, print_every=1, plot_every=1, exp_name=exp_name)


if __name__=="__main__":
    args = options.load_arguments()
    main()
