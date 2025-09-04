 import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load data
@st.cache_data
def load_data():
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    try:
        df = pd.read_csv("diabetes.csv")
        if df.columns.tolist() != cols: df.columns = cols
    except:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        df = pd.read_csv(url, names=cols)
    return df

# Train model
@st.cache_resource
def train_model(df):
    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    return model, scaler, acc

data = load_data()
model, scaler, acc = train_model(data)

# UI
st.title("Diabetes Prediction App")
with st.form("form"):
    c1, c2, c3 = st.columns(3)
    pregnancies = c1.number_input("Pregnancies",0,20,1)
    bp          = c1.number_input("Blood Pressure",0,140,72)
    insulin     = c1.number_input("Insulin",0,900,80)
    glucose     = c2.number_input("Glucose",0,250,120)
    skin        = c2.number_input("Skin Thickness",0,100,20)
    bmi         = c2.number_input("BMI",0.0,80.0,28.0,0.1)
    dpf         = c3.number_input("DPF",0.0,3.0,0.5,0.01)
    age         = c3.number_input("Age",1,120,35)
    submit      = st.form_submit_button("Predict")

if submit:
    inp = pd.DataFrame([{"Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":bp,
                         "SkinThickness":skin,"Insulin":insulin,"BMI":bmi,
                         "DiabetesPedigreeFunction":dpf,"Age":age}])
    st.write("Input:", inp)
    Xs = scaler.transform(inp)
    pred = model.predict(Xs)[0]
    proba = model.predict_proba(Xs)[0]
    label = "Diabetic" if pred else "Not Diabetic"
    conf = proba[pred]
    (st.error if pred else st.success)(f"{label} — {conf*100:.2f}%")
    st.caption(f"Validation Accuracy: {acc*100:.2f}%")
else:
    st.info("Enter details and click Predict")
