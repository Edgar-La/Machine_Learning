import numpy as np
import random as rnd
import pandas as pd

#Open file in readig mode
def abri_arc(nombre_archivo):
  f = open(nombre_archivo, 'r')
  return f
#Save data file into matrix
def datos_a_matriz(f):
  matriz = []
  matriz = [line.split(',') for line in f]
  return matriz
#Close file
def cerrar_archivo(f):
  f.close()

#Save matrix items into a list
def matriz_a_vector(matriz):
	aleatorios = []
	for n in range(len(matriz)):
		for i in range(4):
			aleatorios.append(matriz[n][i])
	return aleatorios

#String to int conversion for every list item
def str_a_int(aleatorios):
  for n in range(len(aleatorios)):
    aleatorios[n] = int(aleatorios[n])
    #print(aleatorios[n])

#List to numpy array conversion
def conver_numpy(aleatorios):
  aleatorios = np.array(aleatorios)
  return aleatorios



###########################################################################################
class Nodo:
	def __init__(self, dato):
		self.left = None
		self.right = None
		self.dato = dato

#Funcion para insertar elementos al arbol
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
f = abri_arc('aleatorios.csv')
matriz = datos_a_matriz(f)
cerrar_archivo(f)
aleatorios = matriz_a_vector(matriz)
str_a_int(aleatorios)
aleatorios = conver_numpy(aleatorios)
print(aleatorios)




root = Nodo(aleatorios[0])
for i in range(1,len(aleatorios)):
	insertar(root,Nodo(aleatorios[i]))

in_order(root)
