import streamlit as st
import pandas as pd
import pickle

loaded_model = pickle.load(open("trained_model.sav", "rb"))
standardised = pickle.load(open("standardised.sav", "rb"))

loaded_diabet_model = pickle.load(open("trained_diabet_model.sav", "rb"))
Standardised_diabet = pickle.load(open("trained_diabet_scalar.sav", "rb"))

loaded_cancer_model = pickle.load(open("breast_cancer_model.sav", "rb"))
standardised_canc = pickle.load(open("breast_cancer_scalar.sav", "rb"))


def calories_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=["Gender","Age","Height","Weight","Duration","Heart_Rate","Body_Temp"])
    
    input_scaled = standardised.transform(input_df)
    prediction = loaded_model.predict(input_scaled)
    result = prediction[0]

    if result < 100:
        return "Light activity", result
    elif result < 250:
        return "Moderate activity", result
    else:
        return "High intensity", result
    
def diabetes_prediction(input_dia_data):
    input_dia_df = pd.DataFrame([input_dia_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    standardised_inp_data = Standardised_diabet.transform(input_dia_df)
    final_pre = loaded_diabet_model.predict(standardised_inp_data)
    if final_pre == 0 :
        print("Not diabetic")
    else :
            print("Diabetic")
    
def breast_cancer_prediction(input_can_data):
    input_data_df = pd.DataFrame([input_can_data] , columns=[
        'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
        'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
        'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
        'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
        ])
    std_data = standardised_canc.transform(input_data_df)
    prediction = loaded_cancer_model.predict(std_data)
    if (prediction[0]==0):
        print('The Breast cancer is Malignant')

    else:
        print('The Breast Cancer is Benign')
    
    
def calories_ui():
    st.title(" Calories Burnt Prediction")

    Gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    Age = st.number_input("Age")
    Height = st.number_input("Height (cm)")
    Weight = st.number_input("Weight (kg)")
    Duration = st.number_input("Duration (min)")
    Heart_Rate = st.number_input("Heart Rate")
    Body_Temp = st.number_input("Body Temperature")

    if st.button("Predict Calories"):
        category, calories = calories_prediction(
            [Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]
        )
        st.success(f"{category} | Calories Burnt: {calories:.2f}")


def diabetes_ui():
    st.title(" Diabetes Prediction")

    Pregnancies = st.number_input("Pregnancies")
    Glucose = st.number_input("Glucose")
    BloodPressure = st.number_input("Blood Pressure")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DPF = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        result = diabetes_prediction([
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DPF, Age
        ])
        st.success(result)


def cancer_ui():
    st.title(" Breast Cancer Prediction")

    columns = [
        'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
        'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
        'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
        'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
    ]

    input_data = []

    for col in columns:
        val = st.number_input(col)
        input_data.append(val)

    if st.button("Predict Cancer"):
        result = breast_cancer_prediction(input_data)
        st.success(f"The tumor is {result}")



def main():
    st.sidebar.title(" Multi-Model Prediction System")

    option = st.sidebar.selectbox(
        "Select Model",
        ["Calories Prediction", "Diabetes Prediction", "Breast Cancer Prediction"]
    )

    if option == "Calories Prediction":
        calories_ui()

    elif option == "Diabetes Prediction":
        diabetes_ui()

    elif option == "Breast Cancer Prediction":
        cancer_ui()


if __name__ == "__main__":
    main()
    
    
