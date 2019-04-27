import csv
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

#This is the tree class
class Test:
	def __init__(self, att_col, att_name, value):
		self.att_col = att_col
		self.att_name = att_name
		self.value = value

	def match(self, instance):
		val = instance[self.att_col]
		if isinstance(val, float):
			return val >= self.value
		else:
			return val == self.value

#Use to recursively build the tree
class Node:
	def __init__(self, test, name, target, children):
		self.test = test
		self.name = name
		self.label, self.cnt = labels(target)
		self.children = children

#Leaf node
class Leaf:
	def __init__(self, value, cnt):
		self.value = value
		self.cnt = cnt

#Function for spliting the data into training and test
def split(data, ratio):
	splitindex = math.ceil(ratio*len(data));
	train = data[0:splitindex]
	test = data[splitindex:]

	return train, test

#Function for filling in empty space in age column
def average_age(data):
	total = 0
	for i in range(len(data)):
		age = data[i][2]
		if age == "":
			total += 0
		else:
			total += float(age)
	return total/len(data)

#Fill in empty space in feature Embarked
def fill_embarked(data):
	S = 0
	C = 0
	Q = 0
	for i in range(len(data)):
		embarked = data[i][-1]
		if embarked == "S":
			S += 1
		elif embarked == "C":
			C += 1
		elif embarked == "Q":
			Q += 1

	if S > Q and S > C:
		fill = "S"
	elif C > S and C > Q:
		fill = "C"
	else:
		fill = "Q"

	for i in range(len(data)):
		if data[i][-1] == "":
			data[i][-1] = fill

#Return unique values of an attribute
def unique(data, col):
	unique = []
	for row in range(len(data)):
		if data[row][col] not in unique:
			unique.append(data[row][col])
	unique.sort()
	return unique

#Helper function for getting unique for target array
def labels(target):
	label = []
	for i in range(len(target)):
		if target[i] not in label:
			label.append(target[i])
	label.sort()

	n = len(label)
	cnt = [0] * (n+1)

	for i in range(len(label)):
		for j in range(len(target)):
			if target[j] == label[i]:
				cnt[i] += 1

	cnt[-1] = len(target)
	return label, cnt

#Find entropy
def entropy(array):
	ent = 0

	for i in range(len(array)-1):
		if array[-1] > 0:
			py = array[i] / array[-1]
			if py > 0:
				ent += -1 * py * math.log(py, 2)
	return ent

#Build a table for calculating conditional entropy
def attribute_table(data, target, attribute_col, t):
	label, count = labels(target)

	col = attribute_col
	n = len(data)

	x_values = unique(data, col)
	y_values = label

	if t is not None:
		x_values = t
		y_values = label

		x_l = 2
		y_l = len(y_values)
	else:
		x_values = unique(data, col)
		y_values = label

		x_l = len(x_values)
		y_l = len(y_values)

	table = [[0 for col in range(y_l+1)] for row in range(x_l)]

	if t is None:
		for i in range(x_l):
			for j in range(y_l):
				for row in range(n):
						if data[row][col] == x_values[i] and target[row] == y_values[j]:
							table[i][j] += 1
	else:
		for j in range(y_l):
			for row in range(n):
					if data[row][col] >= x_values and target[row] == y_values[j]:
						table[0][j] += 1
					elif data[row][col] < x_values and target[row] == y_values[j]:
						table[1][j] += 1


	for i in range(x_l):
		for j in range(y_l):
			table[i][-1] += table[i][j]

	return table

#Find conditional entropy
def conditional_entropy(table):
	row = len(table)
	col = len(table[0])

	total = 0
	for i in range(row):
		total += table[i][-1]

	ent = 0
	for i in range(row):
		px = table[i][-1] / total
		arr = table[i]
		p_condition = entropy(arr)
		ent += px*p_condition

	return ent

#Find info gain
def info_gain(data, target, attribute_col, t):
	label, count = labels(target)
	h_y = entropy(count)
	table = attribute_table(data, target, attribute_col, t)
	h_condition = conditional_entropy(table)

	return h_y - h_condition

#Return a column of a matrix in form of an array
def column(matrix, col):
	return [row[col] for row in matrix]

#Find the best threshold with max info_gain
def find_best_threshold(data, target, col):
	feature = column(data, col)
	feature.sort()

	best_threshold = 0
	max_info = 0
	for i in range(len(feature)-1):
		t = feature[i] + ((feature[i+1] - feature[i])/2)
		info = info_gain(data, target, col, t)
		if info > max_info:
			max_info = info
			best_threshold = t

	return best_threshold, max_info

#for finding the majority target value in dataset
def most_common_label(label, cnt):
	index = 0
	maj = 0
	for i in range(len(cnt)-1):
		val = cnt[i]
		if val > maj:
			maj = val
			index = i
	return label[index]

#id3_helper generate the root node
def id3_helper(data, target, remaining_atts, max_depth):
	label, cnt = labels(target)

	d = deepcopy(data)
	t = deepcopy(target)
	rm = deepcopy(remaining_atts)
	cols_arr = [0] * len(remaining_atts)
	for i in range(len(remaining_atts)):
		cols_arr[i] = i

	child = id3(d, t, rm, cols_arr, max_depth)
	root = Node(None, "root", target, child)

	return root

#The tree building algorithm
#data is the train_data
#target is the train_target
#remaining_atts is just a list of attribute or in this case the header
#cols_arr this hold the position of each attribute with respect to the dataset
#max_depth how big is the tree allow to be
#depth a counter use to track the level of the tree
def id3(data, target, remaining_atts, cols_arr, max_depth, depth=0):
	nodes = []
	label, cnt = labels(target)

	if depth == max_depth:
		value = most_common_label(label, cnt)
		nodes.append(Leaf(value, cnt[int(value)]))
		return nodes
	if len(label) == 1:
		value = label[0]
		nodes.append(Leaf(value, cnt[int(value)]))
		return nodes
	if len(cols_arr) == 0:
		value = most_common_label(label, cnt)
		nodes.append(Leaf(value, cnt[int(value)]))
		return nodes

	ent_y = entropy(cnt)

	max_gain = None
	max_gain_att_col = None

	threshold = None

	x = None

	for i in range(len(cols_arr)):
		if isinstance(data[0][cols_arr[i]], float):
			t, info = find_best_threshold(data, target, cols_arr[i])

			if max_gain is None or info > max_gain:
				max_gain = info
				max_gain_att_col = cols_arr[i]
				threshold = t
				x = i
		else:
			info = info_gain(data, target, cols_arr[i], None)

			if max_gain is None or info > max_gain:
				max_gain = info
				max_gain_att_col = cols_arr[i]
				x = i

	if max_gain is None:
		value = most_common_label(label, cnt)
		nodes.append(Leaf(value, cnt[int(value)]))
		return nodes

	if isinstance(data[0][max_gain_att_col], float):
		values = [threshold]
	else:
		values = unique(data, max_gain_att_col)


	att_name = remaining_atts[max_gain_att_col]
	del cols_arr[x]
	cols = deepcopy(cols_arr)

	for i in range(len(values)):
		test = Test(max_gain_att_col, att_name, values[i])
		if values[0] == threshold:
			greater_than_data = []
			greater_than_target = []
			less_than_data = []
			less_than_target = []
			for j in range(len(data)):
				if test.match(data[j]) == True:
					greater_than_data.append(data[j])
					greater_than_target.append(target[j])
				else:
					less_than_data.append(data[j])
					less_than_target.append(target[j])
			if len(less_than_target) == 0:
				value = most_common_label(label, cnt)
				nodes.append(Leaf(value, cnt[int(value)]))
			else:
				l_child = id3(less_than_data, less_than_target,
										remaining_atts, cols, max_depth, depth+1)
				l_node = Node(test, test.att_name+"<"+str(values[0]),
										less_than_target, l_child)
				nodes.append(l_node)
			if len(greater_than_target) == 0:
				value = most_common_label(label, cnt)
				nodes.append(Leaf(value, cnt[int(value)]))
			else:
				g_child = id3(greater_than_data, greater_than_target,
										remaining_atts, cols, max_depth, depth+1)
				g_node = Node(test, test.att_name+">="+str(values[0]),
										greater_than_target, g_child)
				nodes.append(g_node)
		else:
			subset_data = []
			subset_target = []

			for j in range(len(data)):
				if test.match(data[j]) == True:
					subset_data.append(data[j])
					subset_target.append(target[j])

			if len(subset_target) == 0:
				value = most_common_label(label, cnt)
				nodes.append(Leaf(value, cnt[int(value)]))
			else:
				child = id3(subset_data, subset_target,
										remaining_atts, cols, max_depth, depth+1)
				node = Node(test, test.att_name+"="+values[i], subset_target, child)
				nodes.append(node)

	return nodes

#Function for printing the tree
def print_tree(root, spacing=""):
	if isinstance(root, Leaf):
		print(spacing + "Predict", root.value, root.cnt)
		return

	print(spacing + root.name, root.label, root.cnt)

	for i in range(len(root.children)):
		print(spacing + ' --->: ')
		print_tree(root.children[i], spacing + "  ")

#Function to calculate accuracy of training data
def accuracy_cnt(root):
	if isinstance(root, Leaf):
		return root.cnt

	s = 0
	for i in range(len(root.children)):
		s += accuracy_cnt(root.children[i])

	return s

#Returns a list of predictions
def classify_helper(root, data):
	pred_t = []
	for i in range(len(data)):
		pred_t.append(classify(root.children, data[i]))
	return pred_t

#classify the data point with the tree, excluding the root node
def classify(nodes, instance):
	if isinstance(nodes[0], Leaf):
		return nodes[0].value
	else:
		if isinstance(nodes[0].test.value, float):
			if nodes[0].test.match(instance) == False:
				pred = classify(nodes[0].children, instance)
			else:
				pred = classify(nodes[1].children, instance)
		else:
			for i in range(len(nodes)):
				if nodes[i].test.match(instance) == True:
					pred = classify(nodes[i].children, instance)
	return pred

def test_accuracy(pred_t, actual_t):
	cnt = 0
	for i in range(len(pred_t)):
		if pred_t[i] == actual_t[i]:
			cnt += 1
	accuracy = cnt/len(pred_t)
	return accuracy

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help="select path of dataset as input", default="./data/titanic.csv")

	args = parser.parse_args()

	datafile = args.dataset

	data = []
	target = []

	with open(datafile, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			#Removing Cabin, Ticket, Name, ID column
			del row[10]
			del row[8]
			del row[3]
			del row[0]
			target.append(row[0])
			data.append(row[1:])

	header = data[0]
	data = data[1:]
	target = target[1:]

	avg_age = average_age(data)

	for i in range(len(data)):
		if data[i][2] == "":
			data[i][2] = avg_age
		for j in range(2, 6):
			data[i][j] = float(data[i][j])

	fill_embarked(data)

	#spliting the data and the target
	train_data, test_data = split(data, .6)
	train_target, test_target = split(target, .6)

	#building tree, finding accuracy, ploting
	depths = []
	accuracies_training = []
	accuracies_testing = []
	for i in range(1, len(header)+1): #root is depth = 0
		root = id3_helper(train_data, train_target, header, i)
		accuracy_training = accuracy_cnt(root)/len(train_target)
		pred_test = classify_helper(root, test_data)
		accuracy_testing = test_accuracy(pred_test, test_target)
		depths.append(i)
		accuracies_training.append(accuracy_training)
		accuracies_testing.append(accuracy_testing)


	line1 = plt.plot(depths, accuracies_training, label='Training Data')
	line2 = plt.plot(depths, accuracies_testing, label='Test Data')
	plt.legend()
	plt.ylabel("Accuracy")
	plt.xlabel("Depth")
	plt.title("Accuracy Plot")
	plt.axis([1, 7, 0.6, 1.0])
	plt.yticks(np.arange(0.6, 1.0, step=0.05))
	plt.show()

if __name__ == '__main__':
	main()
