
import sys
import argparse


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--enc_dec_hidden_init", 
            action = "store_true",
            default = False,
            help = "whether to init decoder rnn hidden with encoder rnn hidden, otherwise zeroes"
        )
    argparser.add_argument("--reload_path", 
            default = None,
            help = "path of checkpoint to reload from, None means random init"
        )
    argparser.add_argument("--dec_size", 
            default = 200,
            type=int,
            help = "number of units in decoder rnn"
        )
    argparser.add_argument("--enc_size", 
            default = 200,
            type=int,
            help = "number of units in encoder rnn"
        )
    argparser.add_argument("--enc_layers", 
            default = 1,
            type=int,
            help = "number of layers in encoder rnn"
        )
    argparser.add_argument("--dec_layers", 
            default = 1,
            type=int,
            help = "number of layers in decoder rnn"
        )
    argparser.add_argument("--quick_run", "-q",
            default = False,
            action = "store_true",
            help = "whether to use mini-dataset, so doesn't exceed ram when running locally"
        )
    argparser.add_argument("--mini", "-m",
            default = False,
            action = "store_true",
            help = "whether to use mini-dataset, so doesn't exceed ram when running locally"
        )
    argparser.add_argument("--vgg_layers_to_freeze",
            type = int,
            default = 17,
            help = "how many of vgg19's layers to freeze during training"
        )
    argparser.add_argument("--weight_decay",
            type = int,
            default = 0,
            help = "optimzer"
        )
    argparser.add_argument("--optimizer",
            choices = ['SGD', 'Adam', 'RMS'],
            default = 'Adam',
            help = "optimzer"
        )
    argparser.add_argument("--model",
            choices = ['seq2seq', 'reg', 'eos'],
            default = 'seq2seq',
            help = "which subnetwork to train"
        )
    argparser.add_argument("--verbose",
            action = "store_true",
            default = False,
            help = "whether to print network info before starting training"
        )
    argparser.add_argument("--embedding",
            type = str,
            default = "",
            help = "path to pre-trained VGG weights"
        )
    argparser.add_argument("--save_model",
            type = str,
            default = "",
            help = "path to save model parameters"
        )
    argparser.add_argument("--load_model",
            type = str,
            default = "",
            help = "path to load model"
        )
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data"
        )
    argparser.add_argument("--val",
            type = str,
            default = "",
            help = "path to validation data"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum number of epochs"
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.1,
            help = "dropout probability"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 1e-3,
            help = "learning rate"
        )
    argparser.add_argument("--num_frames",
            type = int,
            default = 8,
            help = "number of frames"
        )
    argparser.add_argument("--frame_width",
            type = int,
            default = 256,
            help = "width of a single frame"
        )
    argparser.add_argument("--frame_height",
            type = int,
            default = 256,
            help = "height of a single frame"
        )
    argparser.add_argument("--ind_size",
            type = int,
            default = 50,
            help = "size of the individuals embeddings"
        )
    argparser.add_argument("--max_length",
            type = int,
            default = 10,
            help = "maximum number of individuals for video"
        )
    argparser.add_argument("--teacher_forcing_ratio",
            type = float,
            default = 1.0,
            help = "teacher forcing ratio"
        )
    argparser.add_argument("--lmbda",
            type = float,
            default = 0.1,
            help = "scalar multiplying the regression loss"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 64,
            help = "number of training examples in each batch"
        )
    argparser.add_argument("--h5_val_file_path",
            type = str,
            default = '/home/eleonora/video_annotation/data/rdf_video_captions/val_50d.h5',
            help = "file to read the data from"
        )
    argparser.add_argument("--h5_train_file_path",
            type = str,
            default = '/home/eleonora/video_annotation/data/rdf_video_captions/train_50d.h5',
            help = "file to read the data from"
        )
    argparser.add_argument("--shuffle",
            #type = bool,
            action = "store_false",
            default = True,
            help = "whether to shuffle that data at each epoch"
        )
    argparser.add_argument("--patience",
            type = int,
            default = 7,
            help = "number of epochs to allow without improvement before early-stopping"
        )
    argparser.add_argument("--output_vgg_size",
            type = int,
            default = 2000,
            help = "size of the output of the vgg layers"
        )





    args = argparser.parse_args()
    return args
