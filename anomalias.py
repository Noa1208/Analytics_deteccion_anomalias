# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# 1. Generar datos sintéticos (simulación de tráfico de red)
12
np.random.seed(42)
normal_traffic = np.random.normal(loc=50, scale=10, size=(1000, 2)) # Datos normales
anomalous_traffic = np.random.normal(loc=100, scale=30, size=(50, 2)) # Datos anómalos
# Combinar datos y crear etiquetas
data = np.vstack([normal_traffic, anomalous_traffic])
labels = np.array([0] * len(normal_traffic) + [1] * len(anomalous_traffic)) # 0 = normal, 1 = anómalo
# Convertir a DataFrame
df = pd.DataFrame(data, columns=["Feature 1", "Feature 2"])
df["Label"] = labels
# 2. Dividir los datos en entrenamiento y prueba
13
X_train, X_test, y_train, y_test = train_test_split(df[["Feature 1", "Feature 2"]],
df["Label"], test_size=0.3, random_state=42)
# 3. Entrenar el modelo Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train)
# 4. Predecir anomalías
y_pred = model.predict(X_test)
y_pred = np.where(y_pred == 1, 0, 1) # Convertir -1 (anómalo) a 1
# 5. Evaluar el modelo
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
# 6. Visualizar los resultados
14
plt.figure(figsize=(10, 6))
# Convertir y_test a array para evitar errores en indexación
y_test = np.array(y_test)
X_test = np.array(X_test)
# Graficar datos normales
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="blue",
label="Normal", edgecolors="k")
# Graficar datos anómalos
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="red",
label="Anómalo", edgecolors="k")
# Graficar predicciones incorrectas
incorrect = y_test != y_pred
15
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c="yellow", label="Error de Predicción", edgecolors="k")
plt.title("Detección de Anomalías en Redes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
# Guardar la imagen en lugar de mostrarla (útil en entornos online)
plt.savefig("grafico_anomalias.png")
plt.show()