import sys
import argparse


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--batch_size", type = int, default = 100)
    argparser.add_argument("--bottleneck", action="store_true", default=False)    
    argparser.add_argument("--classif_size", default = 200, type=int)
    argparser.add_argument("--cuda_visible_devices", type=str, default='0')
    argparser.add_argument("--dataset", type=str, default='MSVD', choices=['MSVD', 'MSRVTT'])
    argparser.add_argument("--device", type=str, default = 'cuda')
    argparser.add_argument("--dropout", type = float, default = 0.1)
    argparser.add_argument("--enc_cnn", choices = ["vgg", "nasnet", "vgg_old"], default = "vgg", help = "which cnn to use as first part of encoder")    
    argparser.add_argument("--enc_init", type=str, choices=['zeroes', 'unit', 'learned', 'unit_learned'], default='unit')
    argparser.add_argument("--enc_layers", default = 2, type=int)
    argparser.add_argument("--enc_rnn", choices = ["gru", "lstm"], default = "gru")
    argparser.add_argument("--enc_size", default = 2000, type=int)
    argparser.add_argument("--exp_name", type=str, default = "")
    argparser.add_argument("--i3d", action="store_true", default=False)    
    argparser.add_argument("--ind_size", type = int, default=300, help="dimensionality of embeddings")
    argparser.add_argument("--learning_rate", type = float, default = 1e-3)
    argparser.add_argument("--lmbda", type=float, default = 1.0)
    argparser.add_argument("--max_epochs", type = int, default = 1000)
    argparser.add_argument("--mini", "-m", action="store_true",help="use dataset of just 6 data points")
    argparser.add_argument("--mlp_size", default = 50, type=int)
    argparser.add_argument("--no_chkpt", action="store_true", help = "don't write checkpoint")    
    argparser.add_argument("--num_frames", type = int, default = 8)
    argparser.add_argument("--ontology", type=str, default='wordnet', choices=['', 'wordnet'])
    argparser.add_argument("--optimizer", choices = ['SGD', 'Adam', 'RMS'], default = 'Adam')
    argparser.add_argument("--output_cnn_size", type = int, default = 4096)
    argparser.add_argument("--overwrite", action="store_true", default=False, help = "whether to overwrite existing experiment of same name")    
    argparser.add_argument("--patience", type = int, default = 7)
    argparser.add_argument("--quick_run", "-q", action="store_true", help="exit training loop after 1 batch")
    argparser.add_argument("--reload", default = None)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--shuffle", action = "store_false", default = True)
    argparser.add_argument("--verbose", action = "store_true", help="print network info before training")
    argparser.add_argument("--weight_decay", type=float, default = 0.0)
        


    args = argparser.parse_args()
    return args


IMPORTANT_PARAMS = [
    'batch_size',
    'bottleneck',
    'cuda_visible_devices',
    'dataset',
    'enc_size', 
    'exp_name', 
    'i3d',
    'ind_size', 
    'learning_rate', 
    'ontology',
    'patience',
    'reload', 
    'seed',
    ]

assert sorted(IMPORTANT_PARAMS) == IMPORTANT_PARAMS, 'Alphabetize your parameters mate'
