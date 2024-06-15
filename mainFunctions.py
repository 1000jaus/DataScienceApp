
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt

# funcion que clasifica las variables de un dataset:
# categoricas
# numericas

def var_classification(df):

    dict = {
        'Categorical': [],
        'Numerical': []
        }
    
    for c in df.columns:
        if df[c].dtype != 'O':
            if len(pd.unique(df[c])) <= 12:
                dict['Categorical'].append(c)
            else:
                dict['Numerical'].append(c)
        else:
            dict['Categorical'].append(c)

    return dict


# función que muestra la distribución de una variable numérica
def plot_hist(df, var, bin_step):
    # Crear el histograma usando Altair
    hist = alt.Chart(df).mark_bar().encode(
        alt.X(var, bin=alt.Bin(maxbins = bin_step), title= var),
        alt.Y('count()', title='Frecuencia')
    ).properties(
        title=f'Frequency Histogram for {var}',
        width=600,
        height=400
    )
    
    # Mostrar el histograma en Streamlit
    st.altair_chart(hist, use_container_width=True)

# función que muestra un pieplot de una variable categórica
def plot_pie(df, var, rad):
    source = pd.DataFrame(df[var].value_counts().reset_index())

    pie = alt.Chart(source).mark_arc(innerRadius=rad).encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field=var, type="nominal")
        ).properties(
        title=f'PiePlot for {var}',
        width=600,
        height=400
        )

    # Mostrar el pieplot en forma de anillo en Streamlit
    st.altair_chart(pie, use_container_width=True)


# función que muestra un scatter de dos variables numéricas
def scatterplot(df, col1, col2):

    #guardar los datos en un dataframe
    data = pd.DataFrame({
        col1: np.array(df[col1]),
        col2: np.array(df[col2]),
    })

    # Crear el scatter plot con Altair
    scatter = alt.Chart(data).mark_circle(size=60).encode(
        x=col1,
        y=col2,
        tooltip=[col1, col2]  # Tooltip para mostrar detalles
    ).properties(
        title = f'ScatterPlot for {col1} and {col2}',
        width=600,
        height=400
    )

    # Mostrar el scatter plot en la aplicación de Streamlit
    st.altair_chart(scatter)