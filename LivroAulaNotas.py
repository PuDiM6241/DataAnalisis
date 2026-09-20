import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

#---------------------DataFrame------------------------#

df = pd.read_csv("data/notas.csv")

#model.fit usa naturalmente o primeiro parametro como uma matriz, linhas sao varios valores para X e colunas adicionam dimensoes.
booksAndClasses = df[["livros", "aulas"]]
grades = df["notas"]

#------------------------------------------------------#

#---------------------Model------------------------#

model = LinearRegression()
model.fit(booksAndClasses, grades)
gradePredict = model.predict(booksAndClasses)

#------------------------------------------------------#



#---------------------Visualize------------------------#

studentsData = pd.DataFrame({
    "livros": [2, 0, 4, 2, 4],
    "aulas": [11, 5, 20, 10, 15]
})

predictions = model.predict(studentsData)

for i in range(len(studentsData)):
    print(
        "Livros:", studentsData["livros"][i],
        "Aulas:", studentsData["aulas"][i],
        "Prediction:", f"{predictions[i]:.3f}"
    )

print("\n\nMODEL")

print("Coef Livros:", model.coef_[0])
print("Coef Aulas:", model.coef_[1])
print("Score:", model.score(booksAndClasses, grades))


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["livros"],
    df["aulas"],
    grades,
    label="Data"
)

x = df["livros"]
y = df["aulas"]
xGrid, yGrid = np.meshgrid(
    np.linspace(x.min(), x.max(), 20),  #faz um espaco com pontos calculado 20 divisoes do menor pro maior x.
    np.linspace(y.min(), y.max(), 20)   #faz um espaco com pontos calculado 20 divisoes do menor pro maior y.
)
zGrid = (
    model.coef_[0] * xGrid
    + model.coef_[1] * yGrid
    + model.intercept_                  #constante
)


ax.plot_surface(
    xGrid,
    yGrid,
    zGrid,
    alpha=0.5
)

ax.set_xlabel("Livros")
ax.set_ylabel("Aulas")
ax.set_zlabel("Nota")
plt.savefig("graphView/LivrosAulasNotasRalation.png")
plt.show()

#------------------------------------------------------#