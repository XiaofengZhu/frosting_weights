import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
sns.set(style="white", context="talk")


def readOneFile(file_path, index1s, index2s):
	with open(file_path, 'r') as f:
		lines = f.readlines()
		value1s = [float(lines[i].strip()) for i in index1s]
		value2s = [float(lines[i].strip()) for i in index2s]
		# lines[index]
	return value1s, value2s

def readAllPairs(file_path):
	value1s, value2s = [], []
	with open(file_path, 'r') as f:
		lines = f.readlines()
		dim = len(lines)
		num_pairs = dim * (dim -1)
		for i in range(dim-1):
			for j in range(i, dim):
				value1s.append(float(lines[i].strip()))
				value1s.append(float(lines[j].strip()))
	return num_pairs, value1s, value2s

# way 2
weight_names = ['weights1_1', 'weights1_2', 'weights3_1', 'weights3_2']
gap = 10
num_docs = 170
# index1s, index2s = list(range(110, 120)), list(range(410, 420))
num_pairs = 0
nums = 0#len(index1s)
for weight_name in weight_names:
	epoch_value1s, epoch_value2s = [], []
	for i in range(130, num_docs):
		num_pairs, value1s, value2s = readAllPairs('corr_{}_output_{}'.format(weight_name, i))
		nums = len(value1s)
		if nums == 0:
			print(weight_name, str(i))
			continue
		epoch_value1s.append(value1s)
		epoch_value2s.append(value2s)

	corr_all = []
	gap_epoch_value1s, gap_epoch_value2s = [], []
	for i in range(0, num_docs-130-1):
		chosen_value1s = epoch_value1s[i]
		gap_epoch_value1s.append(chosen_value1s)
		chosen_value2s = epoch_value2s[i]
		gap_epoch_value2s.append(chosen_value2s)
		if (i + 1) % gap == 0:
			gap_array_1 = np.array(gap_epoch_value1s)
			# np.mean(gap_array, axis = 1)
			gap_array_2 = np.array(gap_epoch_value2s)
			# if len(gap_epoch_value1s) == 0 or len(gap_epoch_value2s) == 0:
			# 	print(weight_name, str(i))
			# else:
			# 	print(gap_epoch_value1s)
			# 	print(gap_epoch_value2s)
			corr_gap = []
			for num in range(nums):
				corr = pearsonr(gap_array_1[:, num], gap_array_2[:, num])
				# print(corr)
				corr_gap.append(corr[0])
			# print(corr_gap)
			corr_all.append(corr_gap)
			gap_array_1 = []		
			gap_array_2 = []
	corr_array = np.array(corr_all)
	larger = (corr_array >0.75).sum()
	smaller = (corr_array < -0.75).sum()
	print('-----------{}---------'.format(weight_name))
	print(nums)
	print(num_pairs)
	# print(corr_array)
	print((larger + smaller) /num_pairs)

# way 1
