import imp
import joblib
import numpy as np
from label import enocode
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Mushrooms Prediction')
st.title('Mushrooms prediction 🍄')
st.markdown('Lets find out your mushrooms are edbible or not')


upload_file=st.file_uploader('Upload your raw .csv file')
if upload_file:    
    df=pd.read_csv(upload_file)
    st.dataframe(df)
    st.header('Your file encoding is done')
    data=enocode(df)
    st.dataframe(data)
    model=joblib.load('mushroom.pkl')
    predict=pd.DataFrame(model.predict(data))
    predict.columns=['Results']
    result=predict.replace({1:'Edible' , 0:'Poisons'})
    st.header('Your prediction is done')
    st.dataframe(result)
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='mushrooms_rescult.csv',
        mime='text/csv',
    )
    st.markdown('-----Created by Avodha Education----')