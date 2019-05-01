import options
import seq2seq
import torch


def main():
	dummy_output = 10
	print(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder = seq2seq.EncoderRNN(args, device).to(device)
	decoder = seq2seq.AttnDecoderRNN(args, device).to(device)

	print("\nENCODER")
	print(encoder)
	print("\nDECODER")
	print(decoder)

	seq2seq.trainIters(args, encoder, decoder, print_every=1000, plot_every=100)


if __name__=="__main__":
    args = options.load_arguments()
    main()