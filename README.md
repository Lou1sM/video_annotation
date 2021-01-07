# Knowledge Graph Extraction from Videos

Code for the paper [Knowledge Graph Extraction from Videos](https://arxiv.org/abs/2007.10040).
Download the word2vec vectors, and place at ../data/w2v_vecs.bin
For each dataset: 
    Download videos and captions (separate files) from Microsoft.
    Preprocess datasets using preprocess_<dataset-name>.py to obtain video tensors
    of the right shape and match with the correct set of captions using a numerical video id.
    Run the VGG and I3D networks, make_vgg_vecs.py and make_i3d_vecs.py to get feature vectors for the videos.
    Train and validate the model using main.py. 
