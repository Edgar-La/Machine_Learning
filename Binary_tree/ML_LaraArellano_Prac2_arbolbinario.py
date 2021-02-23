import numpy as np
import random as rnd
import pandas as pd

#Create a cols x rows matrix with 0-100 random numbers
def random_numbers_matrix(cols, rows):
	matrix = []
	for n in range(rows):
		aux = []
		for i in range(cols):
			aux.append(rnd.randint(0,100))
		matrix.append(aux)
	return matrix

#Save the random numbers matrix into csv file
def matrix_to_csv_file(matrix):
	df = pd.DataFrame(matrix)
	df.to_csv('aleatorios.csv',index = False, header = False)

#Open csv file and save into matrix
def csv_file_to_matrix(name_file):
	df = pd.read_csv(name_file,header = None)
	matrix = np.array(df)
	return matrix

#Fill array with every item in matrix
def matrix_to_vector(matrix):
	aleatorios = []
	for n in range(len(matrix)):
		for i in range(4):
			aleatorios.append(matrix[n][i])
	return np.array(aleatorios)
###########################################################################################
#Create class Node
class Node:
	def __init__(self, dato):
		self.left = None
		self.right = None
		self.dato = dato

#Insert elements into binary tree
def insertar(root, node):
	if root is None:
		root = node
	else:
		if root.dato < node.dato:
			if root.right is None:
				root.right = node
			else:
				insertar(root.right, node)
		else:
			if root.left is None:
				root.left = node
			else:
				insertar(root.left, node)

def in_order(root):
	if root is not None:
		in_order(root.left)
		print(root.dato)
		in_order(root.right)



###########################################################################################
#Calling functions section

#matrix = random_numbers_matrix(4, 8)
#matrix_to_csv_file(matrix)
matrix = csv_file_to_matrix('aleatorios.csv')
aleatorios = matrix_to_vector(matrix)
print(aleatorios)

root = Node(aleatorios[0])
for i in range(1,len(aleatorios)):
	insertar(root,Node(aleatorios[i]))

in_order(root)