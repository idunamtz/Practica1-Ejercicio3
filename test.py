import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def signo(x):
    return np.where(x > 0, 1, -1)

def funcion_activacion(x):
    return 1 / (1 + np.exp(-x))

class PerceptronMulticapa:
    def __init__(self, capas):
        self.pesos = []
        self.bias = []
        for i in range(len(capas) - 1):
            self.pesos.append(np.random.rand(capas[i], capas[i + 1]))
            self.bias.append(np.random.rand(capas[i + 1]))

    def alimentar_adelante(self, x):
        salidas = [x]  # Almacena las salidas de cada capa
        for i in range(len(self.pesos)):
            x = np.dot(x, self.pesos[i]) + self.bias[i]
            x = funcion_activacion(x)
            salidas.append(x)  # Añade la salida de la capa a la lista de salidas
        return x

    def retropropagacion(self, x, d, tasa_aprendizaje):
        salidas = [x]  # Almacena las salidas de cada capa
        for i in range(len(self.pesos)):
            x = np.dot(x, self.pesos[i]) + self.bias[i]
            x = funcion_activacion(x)
            salidas.append(x)  # Añade la salida de la capa a la lista de salidas

        for i in range(len(self.pesos) - 1, -1, -1):
            delta = d * salidas[i + 1] * (1 - salidas[i + 1])  # Aplica función de activación
            dw = np.dot(salidas[i].T, delta)
            self.pesos[i] -= tasa_aprendizaje * dw
            self.bias[i] -= tasa_aprendizaje * delta

            d = np.dot(delta, self.pesos[i].T)

    def entrenar(self, X, d, tasa_aprendizaje, epocas):
        for _ in range(epocas):
            for i in range(len(X)):
                x = X[i]
                y = self.alimentar_adelante(x)

                error = y - d[i]
                self.retropropagacion(x, error, tasa_aprendizaje)

    def clasificar(self, X):
        y = []
        for i in range(len(X)):
            x = X[i]
            y.append(np.argmax(self.alimentar_adelante(x)))
        return y

# Lectura y preprocesamiento del dataset `concentlite.csv`
data = pd.read_csv('concentlite.csv')
X = data[['x1', 'x2']].to_numpy()
y = data['y'].to_numpy()

# Normalización de las características
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Entrenamiento del perceptrón multicapa
capas = [2, 5, 2]  # Ejemplo: 2 neuronas en la entrada, 5 en la capa oculta y 2 en la salida
red = PerceptronMulticapa(capas)
tasa_aprendizaje = 0.01
epocas = 1000
red.entrenar(X_normalized, y, tasa_aprendizaje, epocas)

# Clasificación de nuevos datos
X_nuevo = np.array([[0.1, 0.2], [0.8, 0.9]])
X_nuevo_normalized = scaler.transform(X_nuevo)  # Normaliza los nuevos datos
y_nuevo = red.clasificar(X_nuevo_normalized)
print("Clases de los nuevos datos:", y_nuevo)

# Visualización gráfica
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', label='Datos de entrenamiento')
plt.scatter(X_nuevo[:, 0], X_nuevo[:, 1], c=y_nuevo, cmap='viridis', marker='^', s=100, label='Nuevos datos clasificados')
plt.legend()
plt.title('Clasificación con perceptrón multicapa')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

