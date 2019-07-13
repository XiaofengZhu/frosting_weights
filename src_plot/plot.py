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

def plot(corrs, weight_name, gap=5, metric='Corr'):
	palette = plt.get_cmap('Set1')
	gaps, nums = corrs.shape
	ind = np.arange(gaps)
	epoch_gaps = gap * (ind + 1)
	width = 0.5
	fig = plt.figure(figsize=(8, 6))
	min_v = 1
	max_v = 0
	for num in range(nums):
		min_v = min(min_v, min(corrs[:, num]))
		max_v = max(min_v, max(corrs[:, num]))
		# print(ind)
		# print(corrs[:, num])
		# plt.plot(ind, corrs[:, num], marker='o', \
		# 	markersize=3, color=palette(num), linewidth=1, alpha=0.9, \
		# 	label='Pair {}'.format(num))
		plt.plot(ind, corrs[:, num], marker='o', \
			markersize=3, color=palette(num), linewidth=1, alpha=0.9)		
	plt.ylabel(metric)
	axes = plt.gca()
	axes.set_ylim(-1.0, 1.0)
	# axes.set_ylim(min_v-0.5, max_v+1.5)
	# print(min_v, max_v)
	# plt.xlabel('Other models')
	caption = metric + ' of weight pairs'
	print(caption)
	plt.xticks(ind, epoch_gaps)
	plt.legend(loc='upper left')
	# plt.show()
	fig.savefig(metric + '_{}_{}'.format(weight_name, gaps) + '.pdf')
	plt.close()

weight_names = ['weights1_1', 'weights1_2', 'weights3_1', 'weights3_2']
gap = 10
num_docs = 200
# index1s, index2s = [0, 1, 3, 5, 400], [2, 3, 4, 10, 500]
index1s, index2s = list(range(10)), list(range(10, 20))
for weight_name in weight_names:
	nums = len(index1s)
	epoch_value1s, epoch_value2s = [], []
	for i in range(0, num_docs):
		value1s, value2s = readOneFile('corr_{}_output_{}'.format(weight_name, i), index1s, index2s)
		epoch_value1s.append(value1s)
		epoch_value2s.append(value2s)

	corr_all = []
	gap_epoch_value1s, gap_epoch_value2s = [], []
	for i in range(0, num_docs):
		chosen_value1s = epoch_value1s[i]
		gap_epoch_value1s.append(chosen_value1s)
		chosen_value2s = epoch_value2s[i]
		gap_epoch_value2s.append(chosen_value2s)
		if (i + 1) % gap == 0:
			gap_array_1 = np.array(gap_epoch_value1s)
			# np.mean(gap_array, axis = 1)
			gap_array_2 = np.array(gap_epoch_value2s)
			# print(gap_array_1)
			# print(gap_array_2)
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
	# print(corr_array)
	plot(corr_array, weight_name, gap=gap, metric='Corr')
