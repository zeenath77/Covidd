import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np


data=joblib.load("C:\\Users\\zeena\\OneDrive\\Desktop\\covid\\ml_source\\covid_diag.pkl")
class inp_data(BaseModel):
     Age:int
     Gender:int
     Fever:int
     Cough:int
     Fatigue:int
     Breathlessness:int
     Comorbidity:int
     Stage:int
     Type:int
     Tumor_size:float

app=FastAPI()

@app.get("/")
def root_msg():
     return{"Message": "Welcome to karikalan magic show"}

@app.post("/predict")
def prediction(Data:inp_data):
    #  inp=pd.DataFrame([Data.dict()])
    inp=np.array([[Data.Age,Data.Gender,Data.Fever,Data.Cough,Data.Fatigue,Data.Breathlessness,Data.Comorbidity,Data.Stage,Data.Type,Data.Tumor_size]])
    prdd=data.predict(inp)[0]
    return{"Prediction":prdd}
