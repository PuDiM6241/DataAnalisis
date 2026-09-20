import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

#---------------------DataFrame------------------------#

df = pd.read_csv("data/iris.csv")

#Variaveis usadas para prever a especie
#Muita variedade dimensional para mostrar em grafico
flowerData = df[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]

species = df["species"]

#------------------------------------------------------#

#---------------------Model------------------------#

model = KNeighborsClassifier(n_neighbors=3) #importante ser impar para nao precisar se preucupar com desinpates

model.fit(flowerData, species)

#------------------------------------------------------#

#---------------------Predict------------------------#

#TEST DATA
flowersDataTest = pd.DataFrame({
    "sepal_length": [5, 6, 7.2, 100, 0],
    "sepal_width": [3, 4, 3.8, 100, 0],
    "petal_length": [5, 4.8, 5.2, 100, 0],
    "petal_width": [3, 0.1, 2.0, 100, 0]
})

predictions = model.predict(flowersDataTest)

for i in range(len(flowersDataTest)):
    print(
        "Sepal Length:", flowersDataTest["sepal_length"][i],
        "Sepal Width:", flowersDataTest["sepal_width"][i],
        "Petal Length:", flowersDataTest["petal_length"][i],
        "Petal Width:", flowersDataTest["petal_width"][i],
        "Prediction:", predictions[i]
    )

#------------------------------------------------------#