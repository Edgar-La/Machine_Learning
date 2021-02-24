#Lara Arellano Edgar	|	Feb 23, 2021
#Binary Trees & Traversed algorithms

import numpy as np
import random as rnd
import pandas as pd
#import os; os.system('clear')

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
def matrix_to_csv_file(matrix, name_file):
	df = pd.DataFrame(matrix)
	df.to_csv(name_file,index = False, header = False)

#Open csv file and save into matrix
def csv_file_to_matrix(name_file):
	df = pd.read_csv(name_file,header = None)
	matrix = np.array(df)
	return matrix

#Fill array with every item in matrix
def matrix_to_vector(matrix, cols):
	aleatorios = []
	for n in range(len(matrix)):
		for i in range(cols):
			aleatorios.append(matrix[n][i])
	return np.array(aleatorios)

#Based in
#1)	A. (Sep 07, 2020). 4 Ways To Traverse Binary Trees (with animations!). DEV Community. 
#	https://dev.to/abdisalan_js/4-ways-to-traverse-binary-trees-with-animations-5bi5
#2) Ortiz, J.(Feb 19, 2020). Crear Función para Insertar un Nodo en un Árbol Binario de Búsqueda
#	https://www.youtube.com/watch?v=r2ZwT07C-pI
###########################################################################################
#Create class Node
class Node:
	def __init__(self, dato):
		self.left = None
		self.right = None
		self.dato = dato

#Insert elements into binary tree
def insert(root, node):
	if root is None:
		root = node
	else:
		if root.dato < node.dato:
			if root.right is None:
				root.right = node
			else:
				insert(root.right, node)
		else:
			if root.left is None:
				root.left = node
			else:
				insert(root.left, node)
#In-order algorithm
def in_order(node, in_order_list):
	if node == None:
		return
	in_order(node.left, in_order_list)
	#print(node.dato)
	in_order_list.append(node.dato)
	in_order(node.right, in_order_list)

#Pre-order algorithm
def pre_order(node, pre_order_list):
	if node == None:
		return
	#print(node.dato)
	pre_order_list.append(node.dato)
	pre_order(node.left, pre_order_list)
	pre_order(node.right, pre_order_list)

#Post-order algorithm
def post_order(node, post_order_list):
	if node == None:
		return
	post_order(node.left, post_order_list)
	post_order(node.right, post_order_list)
	#print(node.dato)
	post_order_list.append(node.dato)
###########################################################################################

def traversed_lists_to_csv_file(in_order_list, pre_order_list, post_order_list):
	taversed_lists = np.array([in_order_list, pre_order_list, post_order_list])
	df_traversed = pd.DataFrame(taversed_lists)
	df_traversed.to_csv('recorridos.csv', index = False, header = False)


#------------------------  Calling functions section  -----------------------------

#If you want a new random numbers file, run this 2 lines
#matrix = random_numbers_matrix(4, 8)
#matrix_to_csv_file(matrix, 'aleatorios.csv')

#name indicates what this 2 lines does
matrix = csv_file_to_matrix('aleatorios.csv')
aleatorios = matrix_to_vector(matrix, 4)


#Flll the tree with values
root = Node(aleatorios[0])
for i in range(1,len(aleatorios)):
	insert(root,Node(aleatorios[i]))

#Run traversed algorithms
in_order_list = []; pre_order_list = []; post_order_list = []
in_order(root, in_order_list)
pre_order(root, pre_order_list)
post_order(root, post_order_list)

#the name indicates
traversed_lists_to_csv_file(in_order_list, pre_order_list, post_order_list)


# Verification section
print('\nRandom numbers:'); print(aleatorios)
print('\nIn order:'); print(np.array(in_order_list))
print('\nPre order:'); print(np.array(pre_order_list))
print('\nPost order:'); print(np.array(post_order_list))