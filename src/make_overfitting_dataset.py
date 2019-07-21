import json
import numpy as np
import h5py
from convert_json_to_h5 import padded_emb_seq_from_lists, make_eos_gt


def make_dataset_from_ids(json_file_path, out_h5_file_path, video_ids):

    print('Reading json...')
    with open(json_file_path, 'r') as infile:
        raw_embeddings_and_ids = json.load(infile)
    selected_embeddings_and_ids = [item for item in raw_embeddings_and_ids if item['videoId'] in video_ids]
    N = len(selected_embeddings_and_ids)
    embedding_size = len(selected_embeddings_and_ids[0]['embeddings'][0][0])
    max_len = max([len(dp['embeddings']) for dp in selected_embeddings_and_ids])
    print("Detected embedding size:", embedding_size)
    print("Detected max sequence length:", max_len)

    h5_data = h5py.File(out_h5_file_path, 'w')
    id_dataset = h5_data.create_dataset('videoId', (N,), dtype='uint32')
    emb_seq_dataset = h5_data.create_dataset('embeddings', (N,max_len,embedding_size), dtype=np.float64)
    seq_len_dataset = h5_data.create_dataset('embedding_len', (N,), dtype='uint8')
    eos_gt_dataset = h5_data.create_dataset('eos_gt', (N,max_len), dtype='uint8')

    for idx, dp in enumerate(selected_embeddings_and_ids):
        id_dataset[idx] = dp['videoId']
        list_of_lists = dp['embeddings']
        seq_len_dataset[idx] = len(list_of_lists)
        eos_gt_dataset = make_eos_gt(len(list_of_lists), max_len=max_len)
        emb_seq_dataset[idx] = padded_emb_seq_from_lists(list_of_lists, embedding_size=embedding_size, max_len=max_len)

    h5_data.close()


if __name__ == "__main__":

    video_ids = [1218,1337,1571,1443,1833,1874]

    make_dataset_from_ids(
        json_file_path='/home/eleonora/video_annotation/data/rdf_video_captions/50d.json', 
        out_h5_file_path='../data/rdf_video_captions/50d.{}dp.h5'.format(len(video_ids)),
        video_ids=video_ids
        )
