import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
sns.set(style="white", context="talk")


def readOnce(file_path):
	with open(file_path, 'r') as f:
		lines = f.readlines()
		num_pairs = dim * (dim -1)
	return num_pairs

def readAllPairs(file_path):
	value1s, value2s = [], []
	with open(file_path, 'r') as f:
		lines = f.readlines()
		dim = len(lines)
		for i in range(dim-1):
			for j in range(i, dim):
				value1s.append(float(lines[i].strip()))
				value1s.append(float(lines[j].strip()))
	return value1s, value2s

# way 2
weight_names = ['weights3_2']
di = {}
gap = 40
num_docs = 170
init_epoch = 130
num_pairs = 0
nums = 0#len(index1s)
for weight_name in weight_names:
	num_pairs = readAllPairs('corr_{}_output_{}'.format(weight_name, init_epoch))
	if num_pairs == 0:
		print('nums is 0 ', weight_name)
		continue	
	di[weight_name] = num_pairs
	epoch_value1s, epoch_value2s = [], []
	for i in range(init_epoch, num_docs):
		value1s, value2s = readAllPairs('corr_{}_output_{}'.format(weight_name, i))
		nums = len(value1s)
		epoch_value1s.append(value1s)
		epoch_value2s.append(value2s)

	# corr_all = []
	# gap_epoch_value1s, gap_epoch_value2s = [], []
	# for i in range(0, num_docs-130):
	# 	chosen_value1s = epoch_value1s[i]
	# 	gap_epoch_value1s.append(chosen_value1s)
	# 	chosen_value2s = epoch_value2s[i]
	# 	gap_epoch_value2s.append(chosen_value2s)
	# 	if (i + 1) % gap == 0:
	# 		gap_array_1 = np.array(gap_epoch_value1s)
	# 		# np.mean(gap_array, axis = 1)
	# 		gap_array_2 = np.array(gap_epoch_value2s)
	# 		# if len(gap_epoch_value1s) == 0 or len(gap_epoch_value2s) == 0:
	# 		# 	print(weight_name, str(i))
	# 		# else:
	# 		# 	print(gap_epoch_value1s)
	# 		# 	print(gap_epoch_value2s)
	# 		corr_gap = []
	# 		try:
	# 			for num in range(gap_array_1.shape[1]):
	# 				corr = pearsonr(gap_array_1[:, num], gap_array_2[:, num])
	# 				# print(corr)
	# 				corr_gap.append(corr[0])
	# 			# print(corr_gap)
	# 			corr_all.append(corr_gap)
	# 		except:
	# 			print('wrong')
	# 		gap_array_1 = []		
	# 		gap_array_2 = []
	# corr_array = np.array(corr_all)
	# larger = (corr_array >0.75).sum()
	# smaller = (corr_array < -0.75).sum()
	# print('-----------{}---------'.format(weight_name))
	# num_pairs = di[weight_name]
	# print(num_pairs)
	# # print(corr_array)
	# print((larger + smaller) /num_pairs)

# way 1
