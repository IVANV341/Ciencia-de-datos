#Cargar el archivo pacientes.csv, escalar las columnas edad y colesterol, y luego convertirlo al formato solicitado.
import pandas as pd

from sklearn.impute import SimpleImputer
import math
import matplotlib.pyplot as plt  
import joblib

# 1. Cargar el archivo CSV

df = pd.read_csv('pacientes.csv')
#Realizar un grafico de dispersión entre edad y colesterol, coloreando los puntos según el problema_cardiaco
#imprimir el maximo y el minimo de la edad y del colesterol
print("Edad - Max:", df['edad'].max(), "Min:", df['edad'].min())
print("Colesterol - Max:", df['colesterol'].max(), "Min:", df['colesterol'].min())

#imprimir una grafica para ver la disitrucion de los puntos
                                              
plt.scatter(df['edad'], df['colesterol'], c=df['problema_cardiaco'], cmap='viridis')
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.title('Dispersión entre Edad y Colesterol')
plt.colorbar(label='Problema Cardiaco')
plt.show()
                    
#Cargar el modelo de escalado de joblib del archivo scaler.joblib

scaler = joblib.load('scaler.jb')

#Con esta función ce una red neuronal se va a predecir si va a tener problemas cardiacos o no, se va a usar el modelo de escalado para escalar las columnas edad y colesterol 

def predecir_problema_cardiaco(edad, colesterol):
    # Escalar las columnas X1,x2 usando el modelo de escalado cargado
    datos_transformados = scaler.transform([[edad, colesterol]])
    X1 = datos_transformados[0, 0]*2
    X2 = datos_transformados[0, 1]*2
    print("Datos escalados - Edad:", X1, "Colesterol:", X2)
    # Convertir al formato solicitado
    """Compute a forward pass of the network."""
    
    a1 = 1 / (1 + math.exp(-(-2.1 + (0.53 * X1) + (1.7 * X2))))
    a2 = 1 / (1 + math.exp(-(6.3 + (-1.9 * X1) + (1.9 * X2))))
    a3 = 1 / (1 + math.exp(-(2.7 + (-0.50 * X1) + (-1.6 * X2))))
    a4 = 1 / (1 + math.exp(-(-1.8 + (0.29 * X1) + (7.6 * X2))))
    a5 = 1 / (1 + math.exp(-(0.012 + (2.2 * X1) + (3.2 * X2))))
    a6 = 1 / (1 + math.exp(-(-2.5 + (1.8 * X1) + (-1.2 * X2))))
    a7 = 1 / (1 + math.exp(-(-1.8 + (1.8 * a1) + (-0.80 * a2) + (-3.2 * a3) + (1.9 * a4) + (-1.1 * a5) + (2.2 * a6))))
    a8 = 1 / (1 + math.exp(-(-0.62 + (-0.72 * a1) + (-2.1 * a2) + (0.055 * a3) + (-2.2 * a4) + (-1.4 * a5) + (-1.6 * a6))))
    a9 = 1 / (1 + math.exp(-(1.7 + (-1.6 * a1) + (-5.7 * a2) + (3.0 * a3) + (-4.2 * a4) + (-3.1 * a5) + (-3.0 * a6))))
    a10 = 1 / (1 + math.exp(-(-1.3 + (2.0 * a1) + (-0.51 * a2) + (-2.1 * a3) + (2.2 * a4) + (-2.7 * a5) + (3.1 * a6))))
    a11 = 1 / (1 + math.exp(-(-0.83 + (-2.0 * a7) + (0.99 * a8) + (4.0 * a9) + (-2.5 * a10))))
    a12 = 1 / (1 + math.exp(-(-0.40 + (-2.3 * a7) + (1.3 * a8) + (3.3 * a9) + (-2.3 * a10))))
    a13 = math.tanh(2.3 + (-3.6 * a11) + (-3.2 * a12))
    return a13

# Crear un DataFrame con los datos de entrada

edad=int(input("Ingrese la edad: "))
colesterol=int(input("Ingrese el colesterol: "))
resultado = predecir_problema_cardiaco(edad, colesterol)

# Clasificación final
clase = 1 if resultado >= 0 else -1

print("Resultado de la predicción:", resultado)
print("Clase predicha:", clase)
if clase == 1:
    print("El paciente Si presenta problema cardíaco")
else:
    print("El paciente No presenta problema cardíaco")
