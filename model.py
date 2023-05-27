import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import pickle



dataFrame = pd.read_csv("nutritionEdited.csv")
selected_data= dataFrame[[ "name","calories" , "total_fat" , "protein" , "iron" , "calcium" , "sodium" ,"potassium" , "carbohydrate","fiber","vitamin_d","sugars","serving_size" ]]
data = dataFrame[["calories" , "total_fat" , "protein" , "iron" , "calcium" , "sodium" ,"potassium" , "carbohydrate","fiber","vitamin_d","sugars" ]]


dataNew  = pd.DataFrame(normalize(data) , columns=data.columns)
dataNew_array = dataNew.to_numpy()
X = np.array(dataNew_array)


kmeans = KMeans(n_clusters=3 , init="random").fit(X)
yt = kmeans.labels_

# =======================================================================================
# Weightloss Clustring Classification 
dataToggle_weightloss = data.T
weightlosscat = dataToggle_weightloss.iloc[[1,2,7,8]]
weightlosscat = weightlosscat.T 
weightlosscatData = weightlosscat.to_numpy()
X_train_weightloss  = weightlosscatData
y_train_weightloss = yt
X_test_weightloss = X_train_weightloss[:1000]
y_test_weightloss  = y_train_weightloss [:1000]    
test_data_weightloss = data.iloc[:1000]
clf1=RandomForestClassifier(n_estimators=100)
clf1.fit(X_train_weightloss,y_train_weightloss)
y_pred_weightloss=clf1.predict(X_test_weightloss)
# Save pickle file of weightloss Random Forest
with open("WeightlossModel.pickle" , "wb") as file:
    pickle.dump(clf1 , file )


# ===========================================================================================

# ===========================================================================================
# WeightGain Clustring Classification
dataToggle_weightgain = data.T
weightgaincat= dataToggle_weightgain.iloc[[0,1,2,3,4,7,9,10]]

weightgaincat=weightgaincat.T
weightlgaincatData = weightgaincat.to_numpy()


X_train_weightgain  = weightlgaincatData
y_train_weightgain = yt
X_test_weightgain = X_train_weightgain[:1000]
y_test_weightgain  = y_train_weightgain [:1000]

test_data_weightgain = data.iloc[:1000]
clf2=RandomForestClassifier(n_estimators=100)
clf2.fit(X_train_weightgain,y_train_weightgain)
y_pred_weightgain=clf2.predict(X_test_weightgain)

# Save Pickle file for weightgain Random Forest
with open("WeightgainModel.pickle" , "wb") as file:
    pickle.dump(clf2 , file)

#========================================================================================= 

# ========================================================================================
# WeightMaintain Clustring
dataToggle_weightmaintain = data.T
healthycat = dataToggle_weightmaintain.iloc[[1,2,3,4,6,7,9]]
healthycat = healthycat.T 
healthycatData = healthycat.to_numpy()

X_train_weightmaintain  = healthycatData
y_train_weightmaintain = yt
X_test_weightmaintain = X_train_weightmaintain[:1000]
y_test_weightmaintain  = y_train_weightmaintain [:1000]   
test_data_weightmaintain = data.iloc[:1000]

clf3=RandomForestClassifier(n_estimators=100)
clf3.fit(X_train_weightmaintain,y_train_weightmaintain)
y_pred_weightmaintain=clf3.predict(X_test_weightmaintain)
#Save Pickle File For weightmaintain Random Forest
with open("WeightMaintainModel.pickle" , "wb") as file:
    pickle.dump(clf3 , file) 

# =======================================


