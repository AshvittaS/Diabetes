import gradio as gr
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import os

mlflow.set_tracking_uri(f"file:///{os.getcwd()}/mlruns")

with open("Transformation.pkl", "rb") as f:
    pt = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = mlflow.sklearn.load_model("models:/Diabetes_Best_Model/Production")

cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

continuous_cols = [
    "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def predict_diabetes(Gender, Pregnancies, Glucose, BloodPressure, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age):

    data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]], columns=cols)

    # Apply same transformations as training
    data[continuous_cols] = pt.transform(data[continuous_cols])
    data[cols] = scaler.transform(data[cols])

    pred = model.predict(data)[0]
    pronoun = "He" if Gender.lower() == "male" else "She"
    result = f"{pronoun} {'has' if pred == 1 else 'does not have'} diabetes."
    return result

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Pregnancies", precision=0),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age", precision=0)
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🩺 Diabetes Prediction App (MLflow)",
    description="Model served from MLflow Registry. Enter details to predict whether the person has diabetes or not."
)

if __name__ == "__main__":
    interface.launch()
