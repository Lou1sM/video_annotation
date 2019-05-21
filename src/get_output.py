import os
import json
import sys
import utils
import options
import models
import torch
import numpy as np
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range


def get_output(checkpoint_path, input_tensor, target_number_tensor, rnge, mode='seq2seq', device='cuda'):

	#Load trained model 
	checkpoint = torch.load(checkpoint_path)
	encoder = checkpoint['encoder']
	decoder = checkpoint['decoder']

	print(encoder)
	print(decoder)

	# if mode == 'eos':
	# 	num_predictor = checkpoint['eos']
	# elif mode == 'regressor':
	# 	num_predictor = checkpoint['regressor']
	# else:
	# 	print("Wrong input for mode parameter")
	# 	exit()

	num_test_samples = 3
	encoder.batch_size = num_test_samples
	decoder.batch_size = num_test_samples
	#num_predictor.batch_size = num_test_samples

	# Pass input through encoder
	encoder_hidden = encoder.initHidden().to(device)
	encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

	decoder_input = torch.zeros(1, decoder.batch_size, decoder.hidden_size, device=decoder.device).to(device)
	decoder_hidden_0 = encoder_hidden[encoder.num_layers-1:encoder.num_layers]

	print("real number of embeddings in video: ", target_number_tensor)
		
	batch_decoder_output = []
	for b in range(num_test_samples):
		single_dec_input = decoder_input[:, b].view(1, 1, -1)
		decoder_hidden = decoder_hidden_0[:, b].unsqueeze(1)
		single_dec_output = []
		for l in range(target_number_tensor[b].int()):
			decoder_output, decoder_hidden = decoder(input=single_dec_input, input_lengths=torch.tensor([1]), encoder_outputs=encoder_outputs[:, b].unsqueeze(1), hidden=decoder_hidden) #input_lengths=torch.tensor([target_number_tensor[b]])
			print("decoder output shape:", decoder_output.shape)
			single_dec_output.append(decoder_output.squeeze().detach().cpu().numpy().tolist())
			single_dec_input = decoder_output
		print("number of embeddings in video:", len(single_dec_output))
		batch_decoder_output.append({'videoId':str(rnge[0]+b), 'embeddings':single_dec_output})
	return batch_decoder_output


if __name__=="__main__":

	device='cuda'
	checkpoint_path = '/home/eleonora/video_annotation/checkpoints/chkpt_batch3_lr0.001_enc1_dec1_tfratio1.0_wgDecay0.0_Adam.pt'
	rnge = [1,4]
	test_table = video_lookup_table_from_range(rnge[0],rnge[1])

	num_lines = rnge[1] - rnge[0] 

	h5_test_generator = load_data_lookup('/home/eleonora/video_annotation/data/rdf_video_captions/50d_overfitting.h5', video_lookup_table=test_table, batch_size=num_lines, shuffle=False)
	for iter_, training_triplet in enumerate(h5_test_generator):
		input_tensor = training_triplet[0].float().transpose(0,1).to(device)
		target_tensor = training_triplet[1].float().transpose(0,1).to(device)
		target_number = training_triplet[2].float().to(device)

	output = get_output(checkpoint_path, input_tensor, target_number, rnge)
	#print(output)

	with open('output.txt', 'w') as outfile:
		json.dump(output, outfile)
