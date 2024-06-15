import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from mainFunctions import *

# to run script, type in terminal:
# python -m streamlit run main.py



st.write("""
# Basic EDA and preprocessing for your dataset
         
Hope this helps with the first steps in your model!
""")


if 'number_of_rows' not in st.session_state:
    st.session_state['number_of_rows'] = 5


# ========================================================================

# importar df de prueba

df = pd.read_csv('train.csv')

# ========================================================================

# primer show de los datos
st.subheader('Random df')

# ========================================================================

# botones de mostrar mas o menos registros
col1, col2 = st.columns([0.1, 0.5])

# Agregar CSS para reducir el espacio entre botones
st.markdown("""
    <style>
    .stButton button {
        padding: 10; 
        border: 1px solid grey;
    }
    </style>
    """, unsafe_allow_html=True)

with col1:
    if st.button('Show more:', key = 'button_more'):
        st.session_state['number_of_rows'] += 1


with col2:
    if st.button('Show less:', key = 'button_less'):
        st.session_state['number_of_rows'] -= 1
        if st.session_state['number_of_rows'] < 1:
            st.session_state['number_of_rows'] = 1

# =========================================================================

# mostrar el df:
st.write(df.head(st.session_state['number_of_rows']))

# =========================================================================

# categorización de variables en:

#   1. numericas
#   2. categoricas

# clasificación de variables

st.subheader('Univariant Analysis')

# crea el diccionario separando las variables en categoricas y numericas

types = var_classification(df)

if 'type' not in st.session_state:
    st.session_state['type'] = 'Categorical'

column = st.selectbox('Select a column:', types[st.session_state['type']])

def handle_click_univ(new_type):
    st.session_state['type'] = new_type

type_of_column = st.radio('Type of analysis:', ['Categorical', 'Numerical'])
change = st.button('Change', on_click=handle_click_univ, args = [type_of_column], key = 'button_1')

# =========================================================================

# Análisis univariante:
#   1. numericas    --> histograma de distribución y descripción
#   2. categoricas  --> pieplot de los valores

if st.session_state['type'] == 'Categorical':
    if len(types['Categorical']) > 0:
        # Seleccionar el tamaño de los bins
        rad = st.slider("Select Radius", min_value=0, max_value=100, step = 10, value=30)
        plot_pie(df, column, rad)
    else:
        st.write('No categorical variables in your dataset')
else:
    if len(types['Numerical']) > 0:
        # Seleccionar el tamaño de los bins
        bin_step = st.slider("Select nº of bins", min_value=2, max_value= 100, value=40)
        plot_hist(df, column, bin_step)
        st.table(df[column].describe())
    else:
        st.write('No numerical variables in your dataset')

# =========================================================================

# Análisis bivariante:

# if 'tipe_bi' not in st.session_state:
#     st.session_state['type_bi'] = 'Numerical | Numerical'

st.subheader('Bivariant Analysis')

if 'type_bi' not in st.session_state:
    st.session_state['type_bi'] = 'Numerical | Numerical'

def handle_click_bi(new_type):
    st.session_state['type_bi'] = new_type




type_of_bi = st.radio('Type of analysis:', ['Numerical | Numerical',
                                            'Categorical | Numerical',
                                            'Categorical | Categorical'])

change_bi = st.button('Change', on_click=handle_click_bi, args = [type_of_bi], key = 'button_2')

col_bi_1, col_bi_2 = st.columns([0.5, 0.5])

if st.session_state['type_bi'] == 'Numerical | Numerical':
    # declaramos el tipo de columnas que se pueden seleccionar
    with col_bi_1:
        column_bi_1 = st.selectbox('Select a column:', types['Numerical'], key = 'column_bi_1')
    with col_bi_2:
        column_bi_2 = st.selectbox('Select a column:', types['Numerical'], key = 'column_bi_2')
    
    # ScatterPlot
    scatterplot(df, column_bi_1, column_bi_2)

elif st.session_state['type_bi'] == 'Categorical | Numerical':
    # declaramos el tipo de columnas que se pueden seleccionar
    with col_bi_1:
        column_bi_1 = st.selectbox('Select a column:', types['Categorical'], key = 'column_bi_1')
    with col_bi_2:
        column_bi_2 = st.selectbox('Select a column:', types['Numerical'], key = 'column_bi_2')

    #BoxPlot


else:
    # declaramos el tipo de columnas que se pueden seleccionar
    with col_bi_1:
        column_bi_1 = st.selectbox('Select a column:', types['Categorical'], key = 'column_bi_1')
    with col_bi_2:
        column_bi_2 = st.selectbox('Select a column:', types['Categorical'], key = 'column_bi_2')
    
    #ConfussionMatrix







#   1.   numérica | numérica

#   2. categórica | numérica
#   3. categórica | categórica


# =========================================================================



    


