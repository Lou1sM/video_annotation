from __future__ import print_function
import os
import json
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Function computing: 
# - time elapsed 
# - estimated time remaining given the current time and progress %
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('loss.png')
def read_json_file(input_json):
	## load the json file
	file_info = json.load(open(input_json, 'rb'))

	return file_info

def write_info(input_list, count, f_name):
    #####################################
    # This function gets the input_list which is tuple and write it down in th file.
    # In addition, it writes a count on the top of the file
    ####################################
	with open(f_name, 'w') as f:
		f.write(str(count)+"\n")
		for item in input_list:
			f.write(item[0] + "," + item[1]+"\n")
		f.close()

def get_files_ext(f_dir):
	#####################################
	# This function returns a dict which contains name of files as key and
	# their extensions as the value for a given directory
	#####################################
	f_ext_info = {}
	for f_name in os.listdir(f_dir):
		name, ext = os.path.splitext(f_name)
		f_ext_info[name] = ext

	return f_ext_info

def get_list_files(f_dir, f_ext = None):
	#####################################
	# This function returns a list of files with particular extension if available
	#####################################
	f_list = []
	for f_name in os.listdir(f_dir):
		if os.path.isdir(f_dir + "/"+f_name): # this code doesn't explore the child directory, only parent one
			continue
		name, ext = os.path.splitext(f_name)
		ext = ext.replace(".","")
		if f_ext == None or ext == f_ext:
			f_list.append(f_name)

	return f_list

def get_list_folder(f_dir):
	#####################################
	# This function returns a list of files with particular extension if available
	#####################################
	f_list = []
	for f_name in os.listdir(f_dir):
		if os.path.isdir(f_dir + "/"+f_name): # this code doesn't explore the child directory, only parent one
			f_list.append(f_name)
	return f_list

def get_list_download(f_name):
	#####################################
	# This function get the list of files which are [successfully] downloaded
	# The input file has the following format:
	#
	# 6383
	# video450,https://www.youtube.com/watch?v=0JyMWwkIx2c
	# video3128,https://www.youtube.com/watch?v=TDRT-bYRvMI
	# video3129,https://www.youtube.com/watch?v=SZaoC1YcGMw
	#####################################
	f_name_pre = {}
	with open(f_name, 'r') as f:
		f.readline() # ignore the first line
		for line in f:
			v_id = line.split(',')[0] # only keep the video id
			f_name_pre[v_id] = 1

	return f_name_pre
