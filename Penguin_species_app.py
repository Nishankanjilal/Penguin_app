import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)
df.head()
df = df.dropna()
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})
df['sex'] = df['sex'].map({'Male':0,'Female':1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)
@st.cache()
def predict_species(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex): 
  species=model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]]) 
  species=species[0] 
  if species==0:
    return 'Adelie' 
  elif species==1:
    return 'Chinstrap' 
  else:
    return 'Gentoo'
from sklearn import model_selection
st.title('Penguin Species Prediction Program') 
bl=st.slider('Bill Length (mm)',float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))  
bd=st.slider('Bill Depth (mm)',float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))  
fl=st.slider('Flipper Length (mm)',float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))  
bm=st.slider('Body Mass (g)',float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))  
island=st.selectbox('Island',('Biscoe', 'Dream', 'Torgersen')) 
sex=st.selectbox('Sex',('Male','Female')) 
model=st.selectbox('Classifier Model',('Support Vector Machine','Logistic Regression','Random Forest Classifier'))
if st.button("Predict"):
  if model=='Support Vector Machine':
    species=predict_species(svc_model, bl, bd, fl, bm, island, sex)
    score=svc_model.score(X_train, y_train)
  elif model=='Logistic Regression':
    species=predict_species(log_reg, bl, bd, fl, bm, island, sex)
    score=log_reg.score(X_train, y_train)
  else:
    species=predict_species(rf_clf, bl, bd, fl, bm, island, sex)
    score=rf_clf.score(X_train, y_train)
  st.write("Species predicted:", species)
  st.write("Accuracy score of this model is:", score)
