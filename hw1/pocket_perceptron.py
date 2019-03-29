import numpy as np
import csv

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

def pocket_perceptron(matrix, weights, max_iteration, l_rate):
	for iteration in range(max_iteration):
		current_accuracy = accuracy(matrix, weights)

		print("Accuracy: ", current_accuracy)

		if current_accuracy == 1.0:
			break



		for i in range(len(matrix)):
			prediction = predict(matrix[i][:-1],weights)
			error = matrix[i][-1]-prediction
			for j in range(len(weights)):
				weights[j]=weights[j]+(l_rate*error*matrix[i][j])

	return weights


def main():
	rows = []
	w = [0,0,0,0,0,0]

	with open('Breast_cancer_data.csv', 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		next(csvreader) #remove header
		for row in csvreader:
			for i in range(len(row)):
				row[i] = float(row[i])
			row.insert(0, 1)
			rows.append(row)

	a = pocket_perceptron(rows, w, 10, 1.0)
	print("Final Weights: ", a)

if __name__ == '__main__':
	   main()
