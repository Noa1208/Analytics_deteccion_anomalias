import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Cargar los datos
ruta_archivo = "Filtered_Wireshark_Traffic.csv"  # Cambia esto por la ruta de tu archivo
datos = pd.read_csv(ruta_archivo)

# Preprocesar los datos
# Eliminar columnas no numéricas
datos = datos.drop(columns=["source_ip", "destination_ip"], errors="ignore")

# Separar características (X) y variable objetivo (y)
X = datos.drop(columns=["anomaly"])
y = datos["anomaly"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualizar los resultados
plt.figure(figsize=(10, 6))

# Graficar datos normales
plt.scatter(X_test[y_test == 0]["packets_per_second"], X_test[y_test == 0]["latency_ms"],
            c="blue", label="Normal", edgecolors="k")

# Graficar datos anómalos
plt.scatter(X_test[y_test == 1]["packets_per_second"], X_test[y_test == 1]["latency_ms"],
            c="red", label="Anómalo", edgecolors="k")

# Graficar predicciones incorrectas
incorrect = y_test != y_pred
plt.scatter(X_test[incorrect]["packets_per_second"], X_test[incorrect]["latency_ms"],
            c="yellow", label="Error de Predicción", edgecolors="k")

plt.title("Detección de Anomalías en Redes")
plt.xlabel("Packets per Second")
plt.ylabel("Latency (ms)")
plt.legend()
plt.show()