import numpy as np
import sklearn
from sklearn.cluster import KMeans

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', help="select path of dataset as input", default="./data/titanic.csv")

	args = parser.parse_args()

	datafile = args.dataset

	data = []

	with open(datafile, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')

		for row in csvreader:
		

if __name__ == '__main__':
	main()
	
