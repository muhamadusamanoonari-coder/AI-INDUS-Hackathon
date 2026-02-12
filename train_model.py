import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Example dummy dataset
data = {
    "age":[25,45,50,23,60,35,40,29],
    "bmi":[22,30,28,21,35,26,27,24],
    "bp":[120,140,150,110,160,130,135,118],
    "glucose":[90,160,170,85,200,140,150,95],
    "risk":[0,1,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df.drop("risk",axis=1)
y = df["risk"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2
)

model = RandomForestClassifier()
model.fit(X_train,y_train)

joblib.dump(model,"health_model.pkl")

print("Model trained & saved!")
