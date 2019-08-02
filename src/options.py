import sys
import argparse


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    #argparser.add_argument("--attn_type", choices = ["dot", "ff"], default = "dot")
    argparser.add_argument("--batch_size", type = int, default = 100)
    #argparser.add_argument("--cnn_layers_to_freeze", type = int, default = 17, help = "how many of the CNN's layers to freeze during training")
    argparser.add_argument("--dec_layers", default = 2, type=int)
    argparser.add_argument("--dec_rnn", choices = ["gru", "lstm"], default = "gru")
    argparser.add_argument("--device", type=str, default = 'cuda')
    argparser.add_argument("--dropout", type = float, default = 0.1)
    #argparser.add_argument("--log_pred",action="store_true",help = "log in mlp predictions")    
    argparser.add_argument("--dec_size", default = 1500, type=float)
    argparser.add_argument("--enc_cnn", choices = ["vgg", "nasnet", "vgg_old"], default = "vgg", help = "which cnn to use as first part of encoder")    
    argparser.add_argument("--enc_dec_hidden_init", action = "store_true",help = "init decoder rnn hidden with encoder rnn hidden")
    argparser.add_argument("--enc_init", type=str, choices=['zeroes', 'unit', 'learned', 'unit_learned'], default='unit')
    argparser.add_argument("--enc_layers", default = 2, type=int)
    argparser.add_argument("--enc_rnn", choices = ["gru", "lstm"], default = "gru")
    argparser.add_argument("--enc_size", default = 2000, type=float)
    #argparser.add_argument("--eos_reuse_decoder",action="store_true")
    #argparser.add_argument("--eos_sizes", default = [100,40], nargs='+', type=int)
    argparser.add_argument("--exp_name", type=str, default = "")
    argparser.add_argument("--dec_init", type=str, choices=['zeroes','unit','learned','unit_learned'],default='unit')
    argparser.add_argument("--i3d",action="store_true",help = "use i3d vector (before enc rnn)")
    argparser.add_argument("--i3d_after",action="store_true",help = "put i3d after enc rnn")
    argparser.add_argument("--ind_size", type = int, default=10, help="dimensionality of embeddings")
    argparser.add_argument("--learning_rate", type = float, default = 1e-3)
    #argparser.add_argument("--lmbda_eos", type = float, default = 0.0, help = "weight of eos loss")
    argparser.add_argument("--lmbda_norm", type = float, default = 1.0, help = "weight of norm loss")
    #argparser.add_argument("--loss_func", type=str, choices=['mse', 'cos'], default='mse')
    argparser.add_argument("--max_epochs", type = int, default = 1000)
    argparser.add_argument("--mini", "-m", action="store_true",help="use dataset of just 6 data points")
    argparser.add_argument("--neg_pred_weight",type=float,default=1.0, help = "weight to apply to negative prediction scores")    
    argparser.add_argument("--no_chkpt", action="store_true", help = "don't write checkpoint")    
    #argparser.add_argument("--norm_loss", type=str, choices=['mse','relu'], default='mse')
    argparser.add_argument("--norm_threshold", type=float, default=1.0, help = "value below which the norm is penalized")    
    argparser.add_argument("--num_frames", type = int, default = 8)
    argparser.add_argument("--optimizer", choices = ['SGD', 'Adam', 'RMS'], default = 'Adam')
    argparser.add_argument("--output_cnn_size", type = int, default = 4096)
    argparser.add_argument("--overwrite", action="store_true", default=False, help = "whether to overwrite existing experiment of same name")    
    argparser.add_argument("--patience", type = int, default = 7)
    argparser.add_argument("--pred_embeddings_assist",type=float,default=0.0,help = "how much to move the network outputs to gt when predicting")    
    argparser.add_argument("--pred_margin",type=float,default=10.0,help = "margin within which to apply pred loss, ie we use relu(-/+pred-margin) for pos a neg respectively")
    argparser.add_argument("--quick_run", "-q", action="store_true", help="exit training loop after 1 batch")
    #argparser.add_argument("--reg_sizes", nargs='+', type=int, default=[100,40])
    argparser.add_argument("--reload", default = None)
    #argparser.add_argument("--reweight_eos",action="store_true",help = "apply ones loss")
    argparser.add_argument("--setting", choices = ["embeddings", "preds", "eos", "transformer", 'reg', 'embeddings_eos'], default="embeddings", help = "setting to train the NN on")    
    argparser.add_argument("--shuffle", action = "store_false", default = True)
    argparser.add_argument("--sigmoid_mlp",action="store_true",help = "sigmoid activation function at end of mlp")    
    argparser.add_argument("--teacher_forcing_ratio", type = float, default = 1.0)
    #argparser.add_argument("--transformer_heads", type = int, default = 6)
    #argparser.add_argument("--transformer_layers", type = int, default = 1)
    argparser.add_argument("--verbose", action = "store_true", help="print network info before training")
    argparser.add_argument("--weight_decay", type=float, default = 0.0)
        


    args = argparser.parse_args()
    return args


IMPORTANT_PARAMS = [
    #'attn_type',
    'batch_size',
    'dec_init', 
    'dec_layers', 
    'dec_rnn', 
    'dec_size', 
    #'dropout', 
    'enc_cnn', 
    'enc_init', 
    'enc_layers', 
    'enc_size', 
    #'eos_reuse_decoder',
    #'eos_sizes',
    'exp_name', 
    'i3d',
    'ind_size', 
    'learning_rate', 
    #'lmbda_eos', 
    'lmbda_norm', 
    #'log_pred', 
    #'loss_func', 
    'neg_pred_weight', 
    #'norm_loss',
    'norm_threshold', 
    'pred_embeddings_assist', 
    'pred_margin',
    'reload', 
    #'reweight_eos',
    'setting', 
    #'transformer_heads',
    #'transformer_layers',
    'weight_decay',
    ]

assert sorted(IMPORTANT_PARAMS) == IMPORTANT_PARAMS, 'Alphabetize your parameters mate'
