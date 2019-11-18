import sys
import argparse


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    #argparser.add_argument("--attn_type", choices = ["dot", "ff"], default = "dot")
    argparser.add_argument("--batch_size", type = int, default = 100)
    argparser.add_argument("--cuda_visible_devices", type=str, default='0')
    argparser.add_argument("--dataset", type=str, default='MSVD', choices=['MSVD', 'MSRVTT'])
    argparser.add_argument("--ontology", type=str, default='', choices=['', 'wordnet'])
    argparser.add_argument("--dec_layers", default = 2, type=int)
    argparser.add_argument("--dec_init", type=str, choices=['zeroes','unit','learned','unit_learned'],default='unit')
    argparser.add_argument("--dec_rnn", choices = ["gru", "lstm"], default = "gru")
    argparser.add_argument("--device", type=str, default = 'cuda')
    argparser.add_argument("--dropout", type = float, default = 0.1)
    argparser.add_argument("--dec_size", default = 1500, type=int)
    argparser.add_argument("--enc_cnn", choices = ["vgg", "nasnet", "vgg_old"], default = "vgg", help = "which cnn to use as first part of encoder")    
    argparser.add_argument("--enc_dec_hidden_init", action = "store_true",help = "init decoder rnn hidden with encoder rnn hidden")
    argparser.add_argument("--enc_init", type=str, choices=['zeroes', 'unit', 'learned', 'unit_learned'], default='unit')
    argparser.add_argument("--enc_layers", default = 2, type=int)
    argparser.add_argument("--enc_rnn", choices = ["gru", "lstm"], default = "gru")
    argparser.add_argument("--enc_size", default = 2000, type=int)
    argparser.add_argument("--exp_name", type=str, default = "")
    argparser.add_argument("--final_bottleneck", type=int, default = 0)
    argparser.add_argument("--ind_size", type = int, default=10, help="dimensionality of embeddings")
    argparser.add_argument("--learning_rate", type = float, default = 1e-3)
    argparser.add_argument("--lmbda_norm", type = float, default = 1.0, help = "weight of norm loss")
    argparser.add_argument("--max_epochs", type = int, default = 1000)
    argparser.add_argument("--mini", "-m", action="store_true",help="use dataset of just 6 data points")
    argparser.add_argument("--no_chkpt", action="store_true", help = "don't write checkpoint")    
    argparser.add_argument("--norm_threshold", type=float, default=1.0, help = "value below which the norm is penalized")    
    argparser.add_argument("--num_frames", type = int, default = 8)
    argparser.add_argument("--optimizer", choices = ['SGD', 'Adam', 'RMS'], default = 'Adam')
    argparser.add_argument("--output_cnn_size", type = int, default = 4096)
    argparser.add_argument("--overwrite", action="store_true", default=False, help = "whether to overwrite existing experiment of same name")    
    argparser.add_argument("--patience", type = int, default = 7)
    argparser.add_argument("--pred_embeddings_assist",type=float,default=0.0,help = "how much to move the network outputs to gt when predicting")    
    argparser.add_argument("--pred_margin",type=float,default=10.0,help = "margin within which to apply pred loss, ie we use relu(-/+pred-margin) for pos a neg respectively")
    argparser.add_argument("--pred_normalize", action="store_true", default=False, help = "whether to normlize assisted embeddings before feeding to get_pred")
    argparser.add_argument("--quick_run", "-q", action="store_true", help="exit training loop after 1 batch")
    argparser.add_argument("--reload", default = None)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--setting", choices = ["embeddings", "preds", "transformer", 'reg'], default="embeddings", help = "setting to train the NN on")    
    argparser.add_argument("--shuffle", action = "store_false", default = True)
    argparser.add_argument("--teacher_forcing_ratio", type = float, default = 1.0)
    argparser.add_argument("--verbose", action = "store_true", help="print network info before training")
    argparser.add_argument("--weight_decay", type=float, default = 0.0)
        


    args = argparser.parse_args()
    return args


IMPORTANT_PARAMS = [
    'batch_size',
    'cuda_visible_devices',
    'dataset',
    'dec_size', 
    'enc_size', 
    'exp_name', 
    'ind_size', 
    'learning_rate', 
    'lmbda_norm', 
    'norm_threshold', 
    'ontology',
    'patience',
    'pred_embeddings_assist', 
    'pred_margin',
    'pred_normalize',
    'reload', 
    'seed',
    'setting', 
    ]

assert sorted(IMPORTANT_PARAMS) == IMPORTANT_PARAMS, 'Alphabetize your parameters mate'
