import csv
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt

class Test:
	def __init__(self, column, value):
		self.value = value

	def match(self, instance):
		val = instance[self.column]
		if isinstance(val, float):
			return val >= self.value
		else:
			return val == self.value
	

#This is the tree class
class Node:
	def __init__(self, test):
		self.test = test
		self.children = []

	def add_child(self, obj):
		self.children.append(obj)


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
	return unique

#Helper function for getting unique for target array
def labels(target):
	label = []
	for i in range(len(target)):
		if target[i] not in label:
			label.append(target[i])

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

def majority_label(label, cnt):
	index = 0
	maj = 0
	for i in range(len(cnt)-1):
		val = cnt[i]
		if val > maj:
			maj = val
			index = i
	return label[index]

#The skeleton of building the tree, and the main program
def id3_helper(data, target, remaining_atts):
	label, cnt = labels(target)
	majority = majority_label(label, cnt)

	return id3(data, target, remaining_atts, label, cnt, majority)

def id3(data, target, remaining_atts):
	label, cnt = labels(target)

	if len(label) == 1:
		return
	if not remaining_atts:
		return
	
	ent_y = entropy(cnt)

	max_gain = None
	max_gain_attr = None

	threshold = None

	for i in range(len(remaining_atts)):
		if isinstance(data[0][i], float):
			t, info, attr = find_best_threshold(data, target, i)
			
			if max_gain is None or info > max_gain:
				max_gain = info
				max_gain_attr = attr
				threshold = t
		else:
			info = info_gain(data, target, i, None)
		
			if max_gain is None or info > max_gain:
				max_gain = info
				max_gain_attr = attr
	
	if max_gain is None:
		return
	

	return node

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
	

	t, info = find_best_threshold(train_data, train_target, 2)

	print(t, info)

if __name__ == '__main__':
	main()
