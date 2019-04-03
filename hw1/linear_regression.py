#linear regression as binary classification
import numpy as np
import csv
import argparse
import math

def diagonal_plus(matrix):
	d_plus = matrix.copy()
	for i in range(len(matrix[0])):
		if matrix[i][i] != 0.0:
			d_plus[i][i] = 1.0/matrix[i][i]
	return d_plus

def linear_regression(inputs, label):
	a = np.array(inputs)
	b = a.transpose()
	A_0 = np.matmul(b, a)
	eigenvalue_0, eigenvector_0 = np.linalg.eig(A_0)
	
	diag_0 = np.diag(eigenvalue_0)
	diag_plus_0 = diagonal_plus(diag_0)

	eigenvector_0_t = eigenvector_0.transpose()

	t = np.matmul(eigenvector_0, diag_plus_0)
	A_plus_0 = np.matmul(t, eigenvector_0_t)
	
	a = np.array(label)
	b = np.matmul(b, a)

	w = np.matmul(A_plus_0, b)

	return w

def euclidean_distance(instance, weights):
	a = np.array(instance)
	b = np.array(weights)
	dotproduct = a.dot(b)
	numerator = math.fabs(dotproduct)
	sqr_sum = 0.0
	for i in range(len(weights)):
		sqr_sum += weights[i] ** 2
	denominator = math.sqrt(sqr_sum)

	return numerator/denominator

def binary_classification(inputs, weights0, weights1):
	classified_vector = []
	for i in range(len(inputs)):
		instance = inputs[i][:-2]
		d0 = euclidean_distance(instance, weights0)
		d1 = euclidean_distance(instance, weights1)
		#print("D0 = %.4f and D1 = %.4f"%(d0,d1))
		if d0 < d1:
			classified_vector.append(0)
		else:
			classified_vector.append(1)
	return classified_vector

def accuracy(inputs, prediction):
	total_correct = 0.0
	for i in range(len(inputs)):
		if prediction[i] == inputs[i][-1]:
			total_correct += 1.0
	return total_correct/float(len(inputs))

def main(): 
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset',
						help="select path of dataset as input",
						default="./data/Breast_cancer_data.csv")
	args = parser.parse_args()

	datafile = args.dataset
	
	dataset = []

	subset_0 = [] 
	w0 = []
	y0 = []

	subset_1 = []
	w1 = []
	y1 = []

	with open(datafile, 'r') as csvfile: 
		csvreader = csv.reader(csvfile, delimiter=',') 
		next(csvreader) #remove header 
		for row in csvreader: 
			row.insert(0, 1) #bias
			for i in range(len(row)): 
				row[i] = float(row[i])
			
			dataset.append(row.copy())
			if row[-1] == 0.0:
				del row[-1]
				y0.append(row[-1]) #we are using last feature vector as out label vector
				del row[-1]
				subset_0.append(row)
			else:
				del row[-1]
				y1.append(row[-1]) #we are using last feature vector as out label vector
				del row[-1]
				subset_1.append(row)
	

	w0 = linear_regression(subset_0, y0)
	w1 = linear_regression(subset_1, y1)

	prediction_vector =	binary_classification(dataset, w0, w1)
	print(prediction_vector)
	performance = accuracy(dataset, prediction_vector)
	print(performance)

if __name__ == '__main__':
	main()
