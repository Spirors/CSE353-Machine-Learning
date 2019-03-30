import numpy as np
import csv
import argparse

#helper method for computing the condition that if the inner product is <= 0
def predict(inputs, weights):
	threshold = 0.0
	a = np.array(inputs)
	b = np.array(weights)
	total_activation = a.dot(b)
	return 1.0 if total_activation >= threshold else 0.0

#loops through all inputs to check for accuracy
#if accuracy == 0, no missclassification and we can end the perceptron early
def accuracy(matrix, weights):
	total_correct = 0.0
	for i in range(len(matrix)):
		#-1 is there for eliminating the labeling column
		prediction = predict(matrix[i][:-1], weights)
		if prediction == matrix[i][-1]: 
			total_correct+=1.0
	return total_correct/float(len(matrix))

def perceptron(matrix, weights, max_iteration, l_rate):
	w = weights.copy()

	current_accuracy = accuracy(matrix, w)
	print("Initial Accuracy: ", current_accuracy)

	for iteration in range(max_iteration):

		if current_accuracy == 1.0:
			break

		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1],w)
			error = matrix[i][-1]-prediction
			for j in range(len(w)):
				w[j]=w[j]+(l_rate*error*matrix[i][j])
	
		current_accuracy = accuracy(matrix, w)
		print("Accuracy at iteration %d: "%(iteration+1), current_accuracy)

	return w

def pocket_perceptron(matrix, weights, max_iteration, l_rate):
	w = weights.copy()

	current_accuracy = accuracy(matrix, w)
	print("Initial Accuracy: ", current_accuracy)
	
	w_pocket = weights.copy()

	for iteration in range(max_iteration):
		
		if current_accuracy == 1.0:
			break

		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1],w)
			error = matrix[i][-1]-prediction
			for j in range(len(w)):
				w[j]=w[j]+(l_rate*error*matrix[i][j])
		
		current_accuracy = accuracy(matrix, w)
		pocket_accuracy = accuracy(matrix, w_pocket)
		
		if current_accuracy > pocket_accuracy:
			w_pocket = w.copy()
			pocket_accuracy = current_accuracy

		print("Accuracy of w_pocket at iteration %d: "%(iteration+1), pocket_accuracy)

	return w_pocket



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--version', 
						choices=['naive', 'pocket'],
						help="choose between naive or pocket version of perceptron",
						required=True)

	parser.add_argument('--dataset',
						help="select path of dataset as input",
						default="./data/Breast_cancer_data.csv")

	args = parser.parse_args()

	rows = []

	datafile = args.dataset

	with open(datafile, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		next(csvreader) #remove header
		for row in csvreader:
			for i in range(len(row)):
				row[i] = float(row[i])
			row.insert(0, 1)
			rows.append(row)

	w = [0] * len(rows[0][:-1])

	if args.version == "naive":
		a = perceptron(rows, w, 16, 1.0)
		print("Final Weights: ", a)
	if args.version == "pocket":
		a = pocket_perceptron(rows, w, 16, 1.0)
		print("Final Weights: ", a)

if __name__ == '__main__':
	   main()
