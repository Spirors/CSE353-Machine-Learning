#linear regression as binary classification
import numpy as np
import csv

def main(): 

	subset_0 = [] 
	w0 = []
	subset_1 = []
	w1 = []
	with open('Breast_cancer_data.csv', 'r') as csvfile: 
		csvreader = csv.reader(csvfile, delimiter=',') 
		next(csvreader) #remove header 
		for row in csvreader: 
			for i in range(len(row)): 
				row[i] = float(row[i])
				row.insert(0, 1)
			if row[-1] == 0:
				del row[-1]
				subset_0.append(row)
			else:
				del row[-1]
				subset_1.append(row)

	a = np.array(subset_0)
	b = a.transpose()
	A_0 = np.matmul(b, a)

	c = np.array(subset_1)
	d = c.transpose()
	A_1 = np.matmul(d, c)




if __name__ == '__main__':
	main()
