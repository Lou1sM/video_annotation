import csv
import sys

exp_name = sys.argv[1]

csv_results_file_path = '/home/louis/video_annotation/experiments/{}/{}_summary.csv'.format(exp_name, exp_name)

with open(csv_results_file_path, 'r') as csv_results_file:
    dict_reader = csv.DictReader(csv_results_file)
    for row in dict_reader:
        results_row = row


positive_accuracy_half = int(results_row['tphalf'])/(int(results_row['tphalf'])+int(results_row['fnhalf']))
negative_accuracy_half = int(results_row['tnhalf'])/(int(results_row['tnhalf'])+int(results_row['fphalf']))

positive_accuracy = int(results_row['tp'])/(int(results_row['tp'])+int(results_row['fn']))
negative_accuracy = int(results_row['tn'])/(int(results_row['tn'])+int(results_row['fp']))

print('\nWith threshold of 0.5:')
print('\tPositive Accuracy:', positive_accuracy_half)
print('\tNegative Accuracy:', negative_accuracy_half)

print('\nWith threshold of {}:'.format(results_row['thresh']))
print('\tPositive Accuracy:', positive_accuracy)
print('\tNegative Accuracy:', negative_accuracy)
print()
