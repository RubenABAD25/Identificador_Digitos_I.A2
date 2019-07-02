from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Converter import Converter
import numpy as np

class Red:

	def __init__(self, lim = 946, iteraciones = 100):
		self.converter = Converter(lim)
		self.indices = self.converter.getIndices()
		self.data = self.converter.formater()
		self.label_encoder = LabelEncoder()
		self.datos, self.delta = self.data
		self.iteraciones = iteraciones
		self.codificar()
	
	def codificar(self):
		salida = self.label_encoder.fit_transform(self.delta)
		onehot_encoder = OneHotEncoder(sparse=False)
		salida = salida.reshape(len(salida), 1)
		self.onehot_encoded = onehot_encoder.fit_transform(salida)
	
	def entrenar(self):
		x_train, x_test, d_train, d_test = train_test_split(self.datos, self.onehot_encoded, test_size=0.80, random_state=0)
		self.mlp = MLPClassifier(solver = 'lbfgs', activation='logistic', verbose=True, alpha=1e-4, tol=1e-15, max_iter=self.iteraciones, \
		hidden_layer_sizes=(1024, 512, 256, 128, 10))
		self.mlp.fit(self.datos, self.onehot_encoded)
		
		self.predecir(self.datos[945])
		prediccion = (np.argmax(self.mlp.predict(x_test), axis = 1) + 1).reshape(-1, 1)
		print('Matriz de Confusion\n')
		matriz = confusion_matrix((np.argmax(d_test, axis = 1) + 1).reshape(-1, 1), prediccion)
		print(matriz)
		print('\n')
		print(classification_report((np.argmax(d_test, axis = 1) + 1).reshape(-1, 1), prediccion))

	def predecir(self, entrada):
		res = self.mlp.predict(entrada.reshape(1, -1))
		num = (np.argmax(res, axis=1)+1).reshape(-1, 1) # Decodifica el numero.
		aux = []
		matriz = []
		for i in range(32):
			for j in range(i * 32, (i + 1) * 32):
				aux.append(entrada[j])
			matriz.append(aux)
			aux = []
		for i in range(32):
			print(str(matriz[i]).replace(', ', ''))
		print(res, '=>',int(num[0] - 1))
		return int(num[0] - 1)
		
	def getIndices(self):
		return self.indices
