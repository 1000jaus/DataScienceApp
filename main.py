import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from mainFunctions import *
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import *


# to run script, type in terminal:
# python -m streamlit run main.py



st.write("""
# Basic EDA and preprocessing for your dataset
         
Hope this helps with the first steps in your model!
""")


if 'number_of_rows' not in st.session_state:
    st.session_state['number_of_rows'] = 5
    

#==========================================================================

st.title('Drag and Drop CSV File')

# Instrucciones para el usuario
st.write('Please, drag & drop your csv file right below:')

# Elemento de arrastrar y soltar para subir archivos
uploaded_file = st.file_uploader("CSV file here", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        # Si ocurre un error, mostramos un mensaje personalizado en lugar del error completo
        st.title('Ha ocurrido un error')
        st.write('Por favor, vuelva a intentarlo más tarde.')
        st.stop()
else:
    st.write('Dataframe by default will be boston housing')
    df = pd.read_csv('train.csv')

# ========================================================================

# eliminar indexers

df = delete_indexers(df)

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
        # rad = st.slider("Select Radius", min_value=0, max_value=100, step = 10, value=30)
        plot_pie(df, column)
        plot_barchart(df, column)
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


    # boton para mostrar u ocultar un objeto
    if 'show_corr' not in st.session_state:
        st.session_state['show_corr'] = 'hide'

    def handle_click_corr(new_type):
        st.session_state['show_corr'] = new_type

    
    coefs_corr(df, column_bi_1, column_bi_2)
    
    col_corr_1, col_corr_2 = st.columns([0.1, 0.8])

    with col_corr_1:
        change_bi = st.button('Show', on_click=handle_click_corr, args = ['show'], key = 'button_corr_1')

    with col_corr_2:
        change_bi = st.button('Hide', on_click=handle_click_corr, args = ['hide'], key = 'button_corr_2')
    
    correlaciones(df, st.session_state['show_corr'])

elif st.session_state['type_bi'] == 'Categorical | Numerical':

    # declaramos el tipo de columnas que se pueden seleccionar
    with col_bi_1:
        column_bi_1 = st.selectbox('Select Categorical Column:', types['Categorical'], key = 'column_bi_1')
    with col_bi_2:
        column_bi_2 = st.selectbox('Select Numerical Column:', types['Numerical']  , key = 'column_bi_2')
    
    #BoxPlot
    boxplot(df, column_bi_1, column_bi_2)


else:
    # declaramos el tipo de columnas que se pueden seleccionar
    with col_bi_1:
        column_bi_1 = st.selectbox('Select a column:', types['Categorical'], key = 'column_bi_1')
    with col_bi_2:
        column_bi_2 = st.selectbox('Select a column:', types['Categorical'], key = 'column_bi_2')
    
    #Metricas de asociacion
    contingency_metrics(df, column_bi_1, column_bi_2)

    # boton para mostrar u ocultar un objeto
    if 'show_contingency' not in st.session_state:
        st.session_state['show_contingency'] = 'hide'

    def handle_click_cont(new_type):
        st.session_state['show_contingency'] = new_type

    
    col_cont_1, col_cont_2 = st.columns([0.2, 0.8])

    with col_cont_1:
        change_bi = st.button('Show Heatmap', on_click=handle_click_cont, args = ['show'], key = 'button_corr_1')

    with col_cont_2:
        change_bi = st.button('Hide Heatmap', on_click=handle_click_cont, args = ['hide'], key = 'button_corr_2')
    
    #Metricas de asociacion
    contingency_heatmap(df, column_bi_1, column_bi_2, st.session_state['show_contingency'])





# =========================================================================

# PREPROCESSING

# 1.

# Vamos a empezar seleccionando la variable objetivo para poder transformar
# las variables predictoras:


st.subheader("Data Transformation")

if 'type_of_model' not in st.session_state:
    st.session_state['type_of_model'] = 'Regression'


def handle_click_select_model(new_type):
    st.session_state['type_of_model'] = new_type


type_of_model = st.radio('Type of Model:', ['Regression','Classification'])

if type_of_model == 'Regression':
    target = st.selectbox('Select target variable:', list(var_classification(df)['Numerical']), key = 'target')
elif type_of_model == 'Classification':
    target = st.selectbox('Select target variable:', list(var_classification(df)['Categorical']), key = 'target')

st.button('Change', on_click=handle_click_select_model(type_of_model))

st.write("1. Select Target Variable:")



X, y = separate_df_in_X_and_y(df, target)


# 2.

# Ahora vamos a seleccionar los tipos de preprocesados que queremos para 
# nuestros predictores:

# imputador    : media, mediana, mas frecuente, KNN
# normalizador : estandar, minmax
st.write("2. Preprocessing:")

col_trans_1, col_trans_2 = st.columns([0.5, 0.5])

# declaramos el tipo de columnas que se pueden seleccionar
with col_trans_1:
    imputer = st.selectbox('Imputator:', ["mean", "median", "most frequent", "knn"] ,key = 'imputer')
     
with col_trans_2:
    standard = st.selectbox('Standardization:', ["standard", "minmax"]  , key = 'standard')

X_prep = preprocess_df(X, imputer, standard)


# 3.
# Selección de variables. En función de si es clasificación o regresión
# se aplicará un tipo de selección u otro

st.write('3. Feature Selection:')

dict_methods = {
    'Regression'     : ['None', 'Select K Best', 'rfe', 'SelectFromModel(lasso)', 'mrmr Regression'],
    'Classification' : ['None', 'Select K Best', 'rfe', 'SelectFromModel(RF)', 'mrmr Classification']
}


col_select_1, col_select_2 = st.columns([0.5,0.5])

with col_select_1:
    with st.form(key='my_form'):
        fsm           = st.selectbox('Select a Feature Selection Method:', dict_methods[st.session_state['type_of_model']])
        k             = st.selectbox('Select the number of predictors:', list(range(1, X_prep.shape[1]+1)))
        submit_button = st.form_submit_button(label='Submit')

if "form_data" not in st.session_state:
    st.session_state['form_data'] = {}

if 'x_selected_features' not in st.session_state:
    st.session_state['x_selected_features'] = None

def run_x_selected_features():
    st.session_state['x_selected_features'] = feature_selector(fsm, X_prep, y, target, type_of_model, k)


if submit_button:
    # save the form data in dictionary
    current_form_data = {
        "selbox_1": fsm,
        "selbox_2": k
    }
    # execute only if form data has been modified
    if current_form_data != st.session_state.form_data:
        if fsm != None:
            st.session_state.form_data = current_form_data
            run_x_selected_features()

st.write('Preprocessed DataFrame:')  
st.session_state['x_selected_features']

                
                

# Definir una función para imprimir características seleccionadas
#X_select_features = feature_selector(fsm, X_prep, y, st.session_state['type_of_model'], k)

#st.write(st.session_state['x_selected_features'])

# =========================================================================
# Last step: Model performance

X_selected_features = st.session_state['x_selected_features']

if 'model_ranking' not in st.session_state:
    st.session_state['model_ranking'] = False

def show_or_hide_model_ranking():
     st.session_state['model_ranking'] = not  st.session_state['model_ranking']

st.subheader("Model Performance")
st.write("Once your data has been correctly preprocessed, it's time to check the performance on different models:")


st.button('Ranking', on_click=show_or_hide_model_ranking, key = 'button_ranking')


if st.session_state['model_ranking']:
    st.write('nada')
    #model_ranking(st.session_state['model_ranking'], X_selected_features, y, type_of_model)
