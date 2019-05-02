
import sys
import argparse


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

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
    argparser.add_argument("--dec_size",
            type = int,
            default = 256,
            help = "decoder hidden size"
        )
    argparser.add_argument("--ind_size",
            type = int,
            default = 300,
            help = "size of the individuals embeddings"
        )
    argparser.add_argument("--max_length",
            type = int,
            default = 10,
            help = "maximum number of individuals for video"
        )
    argparser.add_argument("--teacher_forcing_ratio",
            type = float,
            default = 0.5,
            help = "teacher forcing ratio"
        )
    argparser.add_argument("--lmbda",
            type = float,
            default = 0.1,
            help = "scalar multiplying the regression loss"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 1,
            help = "number of training examples in each batch"
        )
    argparser.add_argument("--h5_file_path",
            type = str,
            default = 'data/datasets/single_vid.h5',
            help = "file to read the data from"
        )
    argparser.add_argument("--shuffle",
            type = bool,
            default = True,
            help = "whether to shuffle that data at each epoch"
        )
    argparser.add_argument("--patience",
            type = int,
            default = 7,
            help = "number of epochs to allow without improvement before early-stopping"
        )





    args = argparser.parse_args()
    return args
