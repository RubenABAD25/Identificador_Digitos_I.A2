from Utilities import Utilities
import numpy as np

class Converter:

	def __init__(self, lim = 946):
		self.utilities = Utilities()
		self.lim = lim
		self.indices = self.utilities.generate_indices()
		self.n = []
		self.inicioFin = []
		self.datos = []
		self.delta = []

	def formater(self):
		for j, k, l in self.indices:
			self.n.append(j)
			self.inicioFin.append((k,l))

		for i in range(0, self.lim):
			inicio, fin = self.inicioFin[i]
			fila = np.ravel(np.matrix(self.utilities.get_digit(inicio, fin)))
			self.datos.append(fila)
			self.delta.append(self.n[i])
		return (self.datos, self.delta)
		
	def getIndices(self):
		return self.indices
