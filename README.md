# Knowledge Graph Extraction from Videos

Code for the paper [Knowledge Graph Extraction from Videos](https://arxiv.org/abs/2007.10040).

**Steps to reproduce**
- Prepare video tensors
1. Download the video files for [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) and [MSRVTT](https://www.mediafire.com/folder/h14iarbs62e7p/shared).
2. Preprocess each dataset using preprocess_<dataset-name>.py to obtain video tensors of the right shape and match with the correct set of captions using a numerical video id.
3. Run the VGG and I3D networks, make_vgg_vecs.py and make_i3d_vecs.py to get feature vectors for the videos.

- Prepare logical caption datasets
1. Download the word2vec vectors, and place at ../data/w2v_vecs.bin
2. Run, in order, semantic_parser.py, w2v_wn_links.py and make_new_dset.py. These, respectively, convert the natural language captions to logical captions, link the components of the logical captions to wordnet, and form a new dataset from the linked logical captions (ie format the dataset properly and exclude predicates and individuals appearing fewer than 50 times).

- Train and validate the model using main.py. 
