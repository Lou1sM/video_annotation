import pandas as pd
import json
import sys

exp_name = sys.argv[1]

errors_by_object={}

for dset_fragment in ['val', 'train', 'test']:
    with open('../experiments/{}/{}-{}errors.json'.format(exp_name, exp_name, dset_fragment)) as f:
        d=json.load(f)

    for pos_errors_by_id in d['pos_errors'].values():
        for pos_error in pos_errors_by_id:
            s, p, o = pos_error.split()
            for obj in [s,p,o]:
                if obj not in errors_by_object.keys():
                    errors_by_object[obj] = {'subject': 0, 'object': 0, 'predicate': 0, 'total':0}
            try:
                errors_by_object[s]['subject'] += 1
            except KeyError:
                errors_by_object[s]['subject'] = 1

            try:
                errors_by_object[s]['total'] += 1
            except KeyError:
                errors_by_object[s]['total'] = 1

            try:
                errors_by_object[o]['object'] += 1
            except KeyError:
                errors_by_object[o]['object'] = 1

            try:
                errors_by_object[o]['total'] += 1
            except KeyError:
                errors_by_object[o]['total'] = 1

            try:
                errors_by_object[p]['predicate'] += 1
            except KeyError:
                errors_by_object[p]['predicate'] = 1

            try:
                errors_by_object[p]['total'] += 1
            except KeyError:
                errors_by_object[p]['total'] = 1

    #print(errors_by_object.values())
    
with open('../experiments/{}/{}errors_by_obj.json'.format(exp_name, exp_name), 'w') as f:
    json.dump(errors_by_object,f)

df = pd.DataFrame(errors_by_object).T
df.to_csv('../experiments/{}/{}errors_by_obj.csv'.format(exp_name, exp_name))
