from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Converter import Converter
import numpy as np

class Red:

	def __init__(self, lim = 946):
		self.converter = Converter(lim)
		self.data = self.converter.formater()
		self.label_encoder = LabelEncoder()
		self.datos, self.delta = self.data
	
	def codificar(self):
		salida = self.label_encoder.fit_transform(self.delta)
		onehot_encoder = OneHotEncoder(sparse=False)
		salida = salida.reshape(len(salida), 1)
		self.onehot_encoded = onehot_encoder.fit_transform(salida)
	
	def entrenar(self):
	
		x_train, x_test, d_train, d_test = train_test_split(self.datos, self.onehot_encoded, test_size=0.80, random_state=0)
		mlp = MLPClassifier(solver = 'lbfgs', activation='logistic', verbose=True, alpha=1e-4, tol=1e-15, max_iter=1000, \
		hidden_layer_sizes=(1024, 512, 256, 128, 10))
		mlp.fit(self.datos, self.onehot_encoded)
		
		for entrada in self.datos:
			res = mlp.predict(entrada.reshape(1, -1))
			num = (np.argmax(res, axis=1)+1).reshape(-1, 1)
			#print(entrada)
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
		#print('\nPrueba con {',','.join([str(i) for i in entrada],'} => ', res, '=>',(np.argmax(res, axis=1)+1).reshape(-1, 1))
		prediccion = (np.argmax(mlp.predict(x_test), axis = 1) + 1).reshape(-1, 1)
		print('Matriz de Confusion\n')
		matriz = confusion_matrix((np.argmax(d_test, axis = 1) + 1).reshape(-1, 1), prediccion)
		print(matriz)
		print('\n')
		print(classification_report((np.argmax(d_test, axis = 1) + 1).reshape(-1, 1), prediccion))

red = Red()
red.codificar()
red.entrenar()
