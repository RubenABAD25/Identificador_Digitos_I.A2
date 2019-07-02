from RedNeuronal import Red
from tkinter import *
from tkinter import ttk
import tkinter.messagebox as msg
import numpy as np
from Utilities import Utilities

class Ventana(Frame):

	def __init__(self, master = None):
		super().__init__(root)
		self.master = master
		self.red = Red(946, 1000)
		self.coordenadas = []
		self.indices = self.red.getIndices()
		self.utilities = Utilities()
		self.inicializar()

	def inicializar(self):
		self.master.title("Identificador de digitos")
		self.master.resizable(0, 0)
		self.grid(row = 0,column = 0)
		self.matriz()
		
		btnEntrenar = Button(self, text="Entrenar Red", height=3, bg='yellow', fg='blue', highlightbackground='#3E4149', command=self.entrenar)
		btnEntrenar.grid(columnspan = 11, sticky = W + E + N + S,row = 32, column = 0)
		
		btnReiniciar = Button(self, text="Reiniciar Matriz", height=3, bg='red', highlightbackground='#3E4149', command=self.reiniciar)
		btnReiniciar.grid(columnspan = 10, sticky = W + E + N + S,row = 32, column = 11)
		
		btnPredecir = Button(self, text="Predecir Numero", height=3, bg='blue', fg='yellow', highlightbackground='#3E4149', command=self.predecir)
		btnPredecir.grid(columnspan = 11, sticky = W + E + N + S,row = 32, column = 21)
		
		self.cbxData = ttk.Combobox(self, values = [str(i) for i in self.indices])
		self.cbxData.grid(columnspan = 10, row = 33, column = 11)
		self.cbxData.bind('<<ComboboxSelected>>', self.cargaData)
		

	def  entrenar(self):
		self.red.entrenar()

	def predecir(self):
		matriz = self.generarMatriz(32, self.coordenadas)
		numero = np.ravel(np.matrix(matriz))
		prediccion = self.red.predecir(numero)
		msg.showinfo("Resultado", "El Numero que usted dibujo es: " + str(prediccion))
		#for i in range(len(matriz)):
		#	print(matriz[i])

	def matriz(self):
		self.btn = [[0 for x in range(32)] for x in range(32)] 
		for x in range(32):
			for y in range(32):
				self.btn[x][y] = Button(self, bg='white', command=lambda x1=x, y1=y: self.seleccionar(x1,y1))
				self.btn[x][y].grid(column = x, row = y)

	def generarMatriz(self, n, coordenadas):
		matriz = []
		for i in range(n):
			matriz.append([0 for j in range(n)])

		for i in range(len(coordenadas)):
			x, y = coordenadas[i]
			matriz[y][x] = 1
		return matriz

	def seleccionar(self, x, y):
		self.btn[x][y].config(bg = "black")
		self.coordenadas.append((x, y))
		
	def reiniciar(self):
		self.matriz()
		self.coordenadas = []
		
	def cargaData(self, evt):
		self.reiniciar()
		_id = [int(i) for i in (str(self.cbxData.get()).replace(')','').replace(' ','').replace('(','')).split(',')]
		
		data = self.utilities.get_digit(_id[1],_id[2])
		for i in range(len(data)):
			for j in range(len(data[i])):
				if(data[i][j] == 1):
					self.seleccionar(j, i)
			

if __name__ == '__main__':
	root = Tk()
	ventana = Ventana(root)
	root.mainloop()
