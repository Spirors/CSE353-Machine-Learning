import csv
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt

#The Attribute class is the value of the node
#The column_index is the index of the data to be matched
#The values is an array of different states of an attribute 
#(i.e. 1st, 2nd, 3rd in feature pclass)
#If the attribute is continuous then values contains a threshold t at values[0]
class Attribute:
	def __init__(self, column_index, values):
		self.column_index = column_index
		self.values = values
	
	def is_continuous(self):
		#Test for categorical or continuous variable
		if isinstance(self.values[0], float):
			return True
		else:
			return False			

#This is the tree class
class Node:
	def __init__(self, attr):
		self.attr = attr

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

def unique(data, col):
	unique = []
	for row in range(len(data)):
		if data[row][col] not in unique:
			unique.append(data[row][col])
	return unique

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

def entropy(array):
	ent = 0
	
	for i in range(len(array)-1):
		if array[-1] > 0:
			py = array[i] / array[-1]
			if py > 0:
				ent += -1 * py * math.log(py, 2)
	return ent

def attribute_table(data, target, attribute):
	label, count = labels(target)

	col = attribute.column_index
	n = len(data)

	x_values = attribute.values
	y_values = label

	if attribute.is_continuous() == True:
		x_l = 2
		y_l = len(y_values)
	else:
		x_l = len(x_values)
		y_l = len(y_values)
	
	table = [[0 for col in range(y_l+1)] for row in range(x_l)]
	
	if attribute.is_continuous == False:
		for i in range(x_l):
			for j in range(y_l):
				for row in range(n):
						if data[row][col] == x_values[i] and target[row] == y_values[j]:
							table[i][j] += 1
	else:
		for j in range(y_l):
			for row in range(n):
					if data[row][col] >= x_values[0] and target[row] == y_values[j]:
						table[0][j] += 1
					elif data[row][col] < x_values[0] and target[row] == y_values[j]:
						table[1][j] += 1

	
	for i in range(x_l):
		for j in range(y_l):
			table[i][-1] += table[i][j]
	
	return table

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

def info_gain(data, target, attribute):
	label, count = labels(target)
	h_y = entropy(count)
	table = attribute_table(data, target, attribute)
	print(table)
	h_condition = conditional_entropy(table)

	return h_y - h_condition

def column(matrix, col):
	return [row[col] for row in matrix]

def find_best_threshold(data, target, col):
	feature = column(data, col)
	feature.sort()
	
	best_threshold = 0
	max_info = 0
	for i in range(len(feature)-1):
		t = feature[i] + ((feature[i+1] - feature[i])/2)
		a = Attribute(col, [t])
		info = info_gain(data, target, a)
		print(t, info)
		if info > max_info:
			max_info = info
			best_threshold = t
	
	return best_threshold, max_info
		
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
	
	d = [
			[2.1],
			[3.2],
			[4.7],
			[1.1],
			[3.92],
			[7.4]
		]
	t = [1,1,0,1,0,1]

	age_t, age_if = find_best_threshold(d, t, 0)
	print(age_t)
	print(age_if)
	

if __name__ == '__main__':
	main()
