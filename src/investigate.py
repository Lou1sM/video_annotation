import subprocess
import json
import sys
import os

#exp_name = sys.argv[1]
emb_file_path = sys.argv[1]


#emb_file_path = "/home/louis/video_annotation/experiments/{}/{}-val_outputs.txt".format(exp_name, exp_name)
gt_file_path = "/home/louis/video_annotation/data/rdf_video_captions/10d-det.json.neg"
model_file_path = "/home/louis/video_annotation/rrn-models/model-10d-det.state"
print(emb_file_path)
assert os.path.isfile(emb_file_path)
assert os.path.isfile(gt_file_path)
assert os.path.isfile(model_file_path)
data = json.loads(subprocess.Popen(["vc-eval", emb_file_path], stdout=subprocess.PIPE).communicate()[0].decode())
#data = json.loads(subprocess.check_output("./embedding-gen/run-eval.sh %s %s %s" % (gt_file_path, emb_file_path, model_file_path), shell=True))

print(data['avg-probabilities'])
