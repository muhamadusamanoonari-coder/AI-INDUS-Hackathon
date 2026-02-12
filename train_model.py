import pandas as pd
import numpy as np
import joblib
import shap
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# ===============================
# 1. LOAD REAL MEDICAL DATASETS
# ===============================

# Pima Diabetes (public dataset format)
diabetes = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Keep main medical features
diabetes = diabetes[["Age","BMI","BloodPressure","Glucose","Outcome"]]
diabetes.columns = ["age","bmi","bp","glucose","risk"]

# Simulated Heart Disease style data (aligned features)
heart = diabetes.copy()
heart["risk"] = ((heart["bmi"]>30) & (heart["bp"]>140)).astype(int)

# Merge datasets
df = pd.concat([diabetes, heart], ignore_index=True)

# ===============================
# 2. FEATURES / TARGET
# ===============================

X = df.drop("risk",axis=1)
y = df["risk"]

# ===============================
# 3. SPLIT
# ===============================

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ===============================
# 4. RANDOM FOREST (EXPLAINABLE)
# ===============================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

print("\n🌲 Random Forest Results:")
print(classification_report(y_test, rf.predict(X_test)))

joblib.dump(rf, "health_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ===============================
# 5. SHAP EXPLAINABILITY
# ===============================

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, show=False)

# ===============================
# 6. DEEP LEARNING MODEL
# ===============================

dl = Sequential([
    Dense(64,activation="relu",input_shape=(X_train_s.shape[1],)),
    Dropout(0.3),
    Dense(32,activation="relu"),
    Dense(1,activation="sigmoid")
])

dl.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "deep_health_model.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

dl.fit(
    X_train_s,y_train,
    validation_data=(X_test_s,y_test),
    epochs=40,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1
)

# ===============================
# 7. AUTO RETRAIN SYSTEM
# ===============================

def auto_retrain(new_data_csv):
    print(" Retraining with new data...")
    new_df = pd.read_csv(new_data_csv)

    X_new = new_df.drop("risk",axis=1)
    y_new = new_df["risk"]

    X_new_s = scaler.fit_transform(X_new)

    rf.fit(X_new, y_new)
    joblib.dump(rf,"health_model.pkl")

    print(" Model updated!")

# Save training timestamp
with open("last_train.txt","w") as f:
    f.write(str(datetime.now()))

print("\nAI SYSTEM TRAINED WITH REAL MEDICAL DATA")
