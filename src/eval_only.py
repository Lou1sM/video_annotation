import utils
import options
import models
import torch
import torch.nn as nn
from data_loader import load_data_lookup, video_lookup_table_from_range

device='cuda'
args = options.load_arguments()
print('reloading model from {}'.format(args.reload_path))
saved_model = torch.load(args.reload_path)
reloaded_encoder = saved_model['encoder']
reloaded_decoder = saved_model['decoder']
encoder_optimizer = saved_model['encoder_optimizer']
decoder_optimizer = saved_model['decoder_optimizer']
criterion = nn.MSELoss()

val_table = video_lookup_table_from_range(1,4)
val_generator = load_data_lookup('../data/50d_overfitting.h5', video_lookup_table=val_table, batch_size=args.batch_size, shuffle=False)
for iter_, training_triplet in enumerate(val_generator):
    input_tensor = training_triplet[0].float().transpose(0,1).to(device)
    target_tensor = training_triplet[1].float().transpose(0,1).to(device)
    target_number = training_triplet[2].float().to(device)
    video_id = training_triplet[4].item()
    print('videoId', video_id)
    reload_val_loss = models.eval_on_batch("eval_seq2seq", args, input_tensor, target_tensor, target_number, encoder=reloaded_encoder, decoder=reloaded_decoder, dec_criterion=criterion, device=device)
 
