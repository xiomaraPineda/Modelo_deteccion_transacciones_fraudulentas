import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv('datos_financieros.csv')
print("Primeras filas del dataset:")
print(data.head())

print("Valores nulos por columna:")
print(data.isnull().sum())

print("Información del dataset:")
print(data.type.value_counts()) 

type_counts = data["type"].value_counts().reset_index()
type_counts.columns = ["transaction", "quantity"]

figure = px.pie(
    type_counts,
    values="quantity",
    names="transaction",
    hole=0.5,
    title="Tipos de transacciones"
)
figure.show()
print("Gráfica de distribución de tipos de transacciones generada.")

numeric_data = data.select_dtypes(include='number')

correlation = numeric_data.corr()

print(correlation["isFraud"].sort_values(ascending=False))

print(data["type"].unique())


data["type"] = data["type"].astype(str).str.upper().str.strip()

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})

data["isFraud"] = data["isFraud"].map({0: "Sin fraude", 1: "Fraude"})
print(data.head())

print(data["type"].unique())

x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42) 
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

ypred = model.predict(xtest)
precision = precision_score(ytest, ypred, pos_label="Fraude")
recall = recall_score(ytest, ypred, pos_label="Fraude")
f1 = f1_score(ytest, ypred, pos_label="Fraude")
conf_matrix = confusion_matrix(ytest, ypred, labels=["Sin fraude", "Fraude"])

print("Precisión:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1-score:", round(f1, 2))
print("Matriz de confusión:") 
print(conf_matrix)