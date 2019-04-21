import csv
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt

#entropy functions

def split(data, ratio):
	splitindex = math.ceil(ratio*len(data));
	train = data[0:splitindex]
	test = data[splitindex:]

	return train, test


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help="select path of dataset as input", default="./data/titanic.csv")

	args = parser.parse_args()

	datafile = args.dataset
	
	data = []
	target = []

	with open(datafile, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		next(csvreader) #remove header
		for row in csvreader:
			del row[3]
			del row[0]
			target.append(row[0])
			data.append(row)

	#spliting the data and the target
	train_data, test_data = split(data, .6)
	train_target, test_target = split(target, .6)

	


if __name__ == '__main__':
	main()
