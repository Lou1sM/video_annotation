import numpy as np
import os
import csv

pat_dists= []
dists = []
prob_dists = []

with open('../experiments/jade1-0/jade1-0_summary.csv', 'r') as test_file:
    dreader = csv.DictReader(test_file)
    for row in dreader:
        fieldnames = list(row.keys())
        break


with open('all_results.csv', 'w') as all_results_file:
    dwriter = csv.DictWriter(all_results_file, fieldnames=fieldnames)
    dwriter.writeheader()
    best_legit_dist = 100
    best_prob_dist = -1
    best_params = None
    for i in range(1,9):
        for j in range(108):
            #results_file_path = '../experiments/jade-pred{}-{}/jade-pred{}-{}_summary.csv'.format(i,j,i,j)
            results_file_path = '../experiments/jade{}-{}/jade{}-{}_summary.csv'.format(i,j,i,j)
            if not os.path.isfile(results_file_path):
                print("Can't find", results_file_path)
                print("Moving on to next block\n")
                break
            print("Reading from", results_file_path)
            try:
                with open(results_file_path, 'r') as results_file:
                    dreader = csv.DictReader(results_file)
                    for results_row in dreader:
                        dists.append(float(results_row['l2_distance']))
                        print(results_row['neg_pred_weight'])
                        pat_dists.append(float(results_row['pat_distance']))
                        new_prob_dist = float(results_row['avg_pos_prob']) - float(results_row['avg_neg_prob'])
                        prob_dists.append(new_prob_dist)
                        dwriter.writerow(results_row)
                        #if float(results_row['pat_distance']) < best_legit_dist:
                        if new_prob_dist > best_prob_dist:
                            best_params = results_row
                            #best_legit_dist = float(results_row['pat_distance'])
                            best_prob_dist = new_prob_dist
            except Exception as e:
                print(e)
                print(results_file_path)

    for k,v in best_params.items():
        print(k,v)
    print(best_prob_dist)
    #print('dists:', np.mean(dists), np.var(dists))
    #print('pat dists:', np.mean(pat_dists), np.var(pat_dists))
