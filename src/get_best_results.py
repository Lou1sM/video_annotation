import os
import csv

with open('../experiments/jade1-0/jade1-0_summary.csv', 'r') as test_file:
    dreader = csv.DictReader(test_file)
    for row in dreader:
        fieldnames = list(row.keys())
        break


with open('all_results.csv', 'w') as all_results_file:
    dwriter = csv.DictWriter(all_results_file, fieldnames=fieldnames)
    dwriter.writeheader()
    best_legit_f1 = 0
    best_params = None
    for i in range(1,9):
        for j in range(108):
            results_file_path = '../experiments/jade{}-{}/jade{}-{}_summary.csv'.format(i,j,i,j)
            if not os.path.isfile(results_file_path):
                print("Can't find", results_file_path)
                print("Moving on to next block\n")
                break
            print("Reading from", results_file_path)
            with open(results_file_path, 'r') as results_file:
                dreader = csv.DictReader(results_file)
                for results_row in dreader:
                    dwriter.writerow(results_row)
                    if float(results_row['legit_f1']) > best_legit_f1:
                        best_params = results_row
                        best_legit_f1 = float(results_row['legit_f1'])

    print(best_params)
    print(best_legit_f1)
