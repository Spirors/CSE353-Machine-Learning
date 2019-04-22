import csv
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt

#The Feature class is the value of the node
#The column_index is the index of the data to be matched
#The value is the feature, (i.e. 1st, 2nd, 3rd class)
class Feature:
	def __init__(self, column_index, value):
		self.column_index = column_index
		self.value = value
	
	def match(self, instance):
		instance_val = instance[self.column_index]

		#Test for categorical or continuous variable
		if isinstance(instance_val, float):
			#In the case of continous self.value is the threshold
			return instance_val >= self.value
		else:
			#We simply match for categorical value
			return instance_val == self.value

#This is the tree class
class Node:
	def __init__(self, feature):
		self.feature = feature
		self.false_branch
		self.true_branch


def split(data, ratio):
	splitindex = math.ceil(ratio*len(data));
	train = data[0:splitindex]
	test = data[splitindex:]

	return train, test

def average_age(data):
	total = 0
	for i in range(len(data)):
		age = data[i][2]
		if age == "":
			total += 0
		else:
			total += float(age)
	return total/len(data)

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

	#spliting the data and the target
	train_data, test_data = split(data, .6)
	train_target, test_target = split(target, .6)

if __name__ == '__main__':
	main()
