import pickle
import pandas as pd
import streamlit as st
import numpy as np

st.title('iris_webapp')
lr= pickle.load(open('lr_model.pkl','rb'))
kn= pickle.load(open('kn_model.pkl','rb'))
rf= pickle.load(open('rf_model.pkl','rb'))

ml_model=['logreg','knn','rf']
option= st.sidebar.selectbox('select ml model',ml_model)

sl= st.slider('select sepal length range',0.0,10.0)
sw= st.slider('select sepal width range',0.0,10.0)
pl= st.slider('select petal length range',0.0,10.0)
pw= st.slider('select petal width range',0.0,10.0)

test= [[sl,sw,pl,pw]]

st.write('test data')
st.write(test)

if st.button('classify'):
    if option=='logreg':
        st.success(lr.predict(test))
    elif option=='knn':
        st.sucess(kn.predict(test))
    else:
        st.sucess(rf.predict(test))

