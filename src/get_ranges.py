ranges = [(max([max([e[i] for e in dp['gt_embeddings']]) for dp in d]), min([min([e[i] for e in dp['gt_embeddings']]) for dp in d])) for i in range(25)]
means = [sum([sum([e[i] for e in dp['gt_embeddings']])/(len(dp['gt_embeddings']*1970)) for dp in d]) for i in range(25)]
