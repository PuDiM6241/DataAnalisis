import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#---------------------DataFrame------------------------#

df = pd.read_csv("data/aguaAlpes.csv")
#Ponto de ebulicao(F)
#model.fit usa natrualmente o primeiro parametro como uma matriz, linhas sao varias valores para X e colunas adicionam dimensoes.
bpt = df[["BPt"]]
#Pressao(inHg)
pressure= df["Pressure"]

#------------------------------------------------------#

#---------------------Model------------------------#

model = LinearRegression()
model.fit(bpt, pressure)
pressurePredict = model.predict(bpt)

#------------------------------------------------------#



#---------------------Visualize------------------------#

print("P(200F): ", model.predict([[200]])[0])#warning que gera é porque as colunas nao tem nome para ele oredenar, como tem só uma nao tem problema
print("P(215F): ", model.predict([[215]])[0])#warning que gera é porque as colunas nao tem nome para ele oredenar, como tem só uma nao tem problema
print("\n\nMODEL\nCoef:", model.coef_[0])
print("Score:", model.score(bpt, pressure))

plt.scatter(bpt, pressure, label="Data")
plt.plot(bpt, pressurePredict, label="linear Regression")

plt.xlabel("Temperature")
plt.ylabel("Pressure")
plt.savefig("graphView/BPtPressureRelation.png")
plt.show()

#------------------------------------------------------#