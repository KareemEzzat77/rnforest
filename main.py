from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel 
import model
import pickle
import pandas as pd
import numpy as np

app = FastAPI()


class item(BaseModel):
    height:float
    weight:float

@app.post("/")
def index(item:item):
    bmi = item.weight/((item.height/100)**2)
    Food_recommendation =[]
    food_item = {}
    with open("WeightgainModel.pickle" , "rb") as file:
            modelFile=pickle.load(file)
        
    classify=modelFile.predict(model.X_test_weightgain)
    value_return = pd.DataFrame(model.test_data_weightgain[classify==2]) 
    index_values= model.selected_data.iloc[value_return.index].values
    for i in index_values:
         food_item={"Name":i[0] , "Calories":i[1] , "Fat":i[2] , "Protein":i[3] , "Carb":i[8] ,"Amount":i[12]}
         Food_recommendation.append(food_item)
   
    return Food_recommendation
       


    