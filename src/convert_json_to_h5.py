import json
import numpy as np
import h5py


def convert_json_to_h5(json_file_path, out_h5_file_path):
    """Convert json in agreed format to h5 in format expected by the Dataset.

    The json contains a list of datapoint, each having keys 'videoId', 'embed-
    dings', 'individuals' and 'captions'. Only the first two are passed to the
    h5 file. The videoIds are ints in the json file and are passed as is to the
    h5 file. The embeddings are unpadded, unsorted lists of lists in the json
    file. They are converted to np arrays, padded and sorted (by first element)
    before being passed to the h5 file. Their unpadded length is also measured
    and passed to the h5 file as 'embedding_len'. 

    The h5 file thus contains 3 datasets: videoId (ints), embeddings (np arrays)
    and embedding_lens (ints).

    Args:
      json_file_path: path of the json file to read from
      out_h5_file_path: path of the h5 file to write to
    """

    print('Reading json...')
    with open(json_file_path, 'r') as infile:
        embeddings_and_ids = json.load(infile)
    N = len(embeddings_and_ids)
    max_individuals = 10
    h5_f = h5py.File(out_h5_file_path, 'w')
    id_dataset = h5_f.create_dataset('videoId', (N,), dtype='uint32')
    emb_seq_dataset = h5_f.create_dataset('embeddings', (N,max_individuals,300), dtype=np.float64)
    seq_len_dataset = h5_f.create_dataset('embedding_len', (N,), dtype='uint8')
    print('Making h5...')
    for idx, dp in enumerate(embeddings_and_ids):
        new_vid_id = dp['videoId']
        id_dataset[idx] = new_vid_id
        list_of_lists = dp['embeddings']
        seq_len_dataset[idx] = len(list_of_lists)
        emb_seq_dataset[idx] = padded_emb_seq_from_lists(list_of_lists)

    
    h5_f.close()


