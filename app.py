import gradio as gr
import pickle
import numpy as np
import pandas as pd

with open('Transformation.pkl', 'rb') as f:
    pt = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

cols = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

def predict_diabetes(Gender, Pregnancies, Glucose, BloodPressure, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age):
 
    data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]],
                        columns=cols)
    
    con_col = ['Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    data[con_col] = pt.transform(data[con_col])
    data[cols] = scaler.transform(data[cols])
    
    pred = model.predict(data)[0]
    
    pronoun = "He" if Gender.lower() == "male" else "She"

    if pred == 1:
        result = f" {pronoun} has diabetes."
    else:
        result = f" {pronoun} does not have diabetes."
    
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
    title="🩺 Diabetes Prediction App",
    description="Enter the details below to predict whether the person has diabetes or not."
)

if __name__ == "__main__":
    interface.launch()
