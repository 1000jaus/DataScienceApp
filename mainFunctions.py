
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from scipy.stats import *
from sklearn.impute import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel, RFE
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mrmr
import lazypredict
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from mrmr import *
from sklearn.model_selection import train_test_split
import time




# funcion que clasifica las variables de un dataset:
# categoricas
# numericas


def delete_indexers(df):

    for c in ['Id', 'id', 'Index', 'index']:
        if c in df.columns:
            df = df.drop(columns = [c])

    return df
        


def var_classification(df):

    dict = {
        'Categorical': [],
        'Numerical'  : [],
        'Exclude'    : []
        }
    
    num = np.array(df.select_dtypes(include=['int64', 'float64']).columns)
    cat = np.array(df.drop(columns=num).columns)

    # migrate = []

    # for c in num:
    #     if len(np.unique(df[c])) < 20:
    #         migrate.append(c)
    
    # for c in migrate:
    #     index = np.where(num == c)
    #     num = np.delete(num, index)
    #     cat = np.append(cat, c)

    dict['Numerical']   = num
    dict['Categorical'] = cat

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
def plot_barchart(df, var):


    source = pd.DataFrame(df[var].value_counts().reset_index())
    source.columns = [var, 'count']

    pie = alt.Chart(source).mark_arc(innerRadius=70).encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field=var, type="nominal")
        ).properties(
        title=f'PiePlot for {var}',
        width=600,
        height=400
        )

    # Mostrar el pieplot en forma de anillo en Streamlit
    st.altair_chart(pie, use_container_width=True)

    bar = alt.Chart(source).mark_bar().encode(
        x= var + ':O',
        y='count:Q',
        color=alt.Color(field=var, type="nominal")
        ).properties(
        title=f'BarChart for {var}',
        width=600,
        height=400
        )

    # Mostrar el pieplot en forma de anillo en Streamlit
    st.altair_chart(bar, use_container_width=True)


# función que muestra un scatter de dos variables numéricas
def scatterplot(df, col1, col2):

    n_samples = min(df.shape[0], 2000)
    #guardar los datos en un dataframe
    data = pd.DataFrame({
        col1: np.array(df[col1])[0:n_samples],
        col2: np.array(df[col2])[0:n_samples],
    })

    # Crear el scatter plot con Altair
    scatter = alt.Chart(data).mark_circle().encode(
        x = alt.X(col1, scale = alt.Scale( domain = [ min(data[col1]), max(data[col1]) ] )),
        y = alt.Y(col2, scale = alt.Scale( domain = [ min(data[col2]), max(data[col2]) ] )),
        tooltip=[col1, col2]  # Tooltip para mostrar detalles
    ).properties(
        title = f'ScatterPlot for {col1} and {col2}',
        width=600,
        height=400
    )

    # Mostrar el scatter plot en la aplicación de Streamlit
    st.altair_chart(scatter)


def variables_mas_correlacionadas(df):

    # Calcular matriz de correlación
    correlaciones = df.corr().abs()

    # Obtener la matriz triangular inferior de la matriz de correlación
    matriz_triang_inferior = np.tril(correlaciones, k=-1)

    # Obtener las coordenadas de las correlaciones más altas en la matriz triangular inferior
    filas, columnas = np.where(matriz_triang_inferior > 0)

    # Obtener las variables correspondientes a las correlaciones más altas
    variables_mas_correlacionadas = [(correlaciones.index[fila], correlaciones.columns[columna]) for fila, columna in zip(filas, columnas)]

    return variables_mas_correlacionadas


def correlaciones(df, action):

    if action == 'show':

        num_cols = var_classification(df)['Numerical']
        df_num = df[num_cols]

        correlaciones = df_num.corr().abs()
        
        for i in range(0,len(correlaciones[0:])):
            for j in range(i):
                correlaciones.iloc[i,j] = 0

        # Obtener variables más correlacionadas
        correlacion_max = correlaciones.unstack().sort_values(ascending=False)

        # Eliminar pares de variables duplicadas y la correlación consigo misma
        correlacion_max = correlacion_max[correlacion_max.index.get_level_values(0) != correlacion_max.index.get_level_values(1)]
        
        st.write('Most correlated variables ranking:')
        for i, (variables, correlacion) in enumerate(correlacion_max.items()):
            if i == 5:  # Mostrar solo las 5 primeras variables
                break
            st.write(f"{i+1}. {variables[0]} y {variables[1]}: {correlacion:.2f}")

        st.write(correlaciones)

    else:
        st.write('')


def coefs_corr(df, col1, col2):
    
    arr1 = np.array(list(df[col1].values))
    arr2 = np.array(list(df[col2].values))

    arr1, arr2 = arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)]

    max_len = min([len(arr1), len(arr2)])

    arr1, arr2 = arr1[0:max_len], arr2[0:max_len]


    coefs_desc = ["Pearson",
                  "Spearman",
                  "Kendall's Tau",
                  "R²"]

    # Calcular el coeficiente de Pearson
    pearson_corr, _ = pearsonr(arr1, arr2)

    # Calcular el coeficiente de Spearman
    spearman_corr, _ = spearmanr(arr1, arr2)

    # Calcular el coeficiente de Tau de Kendall
    kendall_tau, _ = kendalltau(arr1, arr2)
    
    # Calcular la regresión lineal
    slope, intercept, r_value, _, _ = linregress(arr1, arr2)

    # Calcular el coeficiente de determinación (R²)
    coefficient_of_determination = r_value**2

    coefs_res = [pearson_corr,
                 spearman_corr,
                 kendall_tau,
                 coefficient_of_determination]
    
    data = pd.DataFrame({
        'coefs': np.array(coefs_desc),
        'value': np.array(coefs_res),
    })

    st.write('Correlation values:')
    st.write(data)


# función que muestra un scatter de dos variables numéricas
def boxplot(df, cat, num):

    # Crear el boxplot
    boxplot = alt.Chart(df).mark_boxplot(color = 'grey').encode(
        alt.X(cat + ':N'),
        alt.Y(num + ':Q', scale = alt.Scale( domain = [ min(df[num]), max(df[num]) ])),
        alt.Color(cat + ":N").legend(None)
    ).properties(
        title = f'Boxplot de {num} por {cat}',
        width=600,
        height=600
    )

    # Mostrar el scatter plot en la aplicación de Streamlit
    st.altair_chart(boxplot)

def boxplot1(df, col1, col2):
    print(col1, col2)


def contingency_metrics(df, cat1, cat2):

    # Contingency table
    contingency_table = pd.crosstab(df[cat1], df[cat2])

    st.write('1. Contingency_table')
    st.write(contingency_table)
    # Prueba chi-cuadrado
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Tamaño de la muestra
    n = contingency_table.sum().sum()

    # Coeficiente de Contingencia
    contingency_coefficient = np.sqrt(chi2 / (chi2 + n))

    # Índice de Cramer (Cramer's V)
    r, k = contingency_table.shape
    cramers_v = np.sqrt(chi2 / (n * min(k-1, r-1)))

    coefs_desc = ["Chi-2",
                  "P-value",
                  "Contingency Coef.",
                  "Cramer's V"]
    
    coefs_res = [chi2,
                p,
                contingency_coefficient,
                cramers_v]
    
    st.write('2. Test Chi-2:')

    data = pd.DataFrame({
        'coefs': np.array(coefs_desc)[0:2],
        'value': np.array(coefs_res)[0:2],
    })

    # Interpretación
    alpha = 0.05
    if p < alpha:
        st.markdown('<p style="color: green;">Reject the null hypothesis. There is a significant association between the variables.</p>', 
                    unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: red;">Do not reject the null hypothesis. There is no significant association between the variables.</p>', 
            unsafe_allow_html=True)
    st.write(data)

    # resto de métricas
    data = pd.DataFrame({
        'coefs': np.array(coefs_desc)[2:4],
        'value': np.array(coefs_res)[2:4],
    })

    st.write('3. Association Metrics:')
    st.write(data)


def contingency_heatmap(df, cat1, cat2, state):

    if state == 'hide':
        st.write('')

    else:
        # Contingency table
        contingency_table = pd.crosstab(df[cat1], df[cat2]).reset_index()

        if cat1 == cat2:
            cat2 += ' '
        

        contingency_table_long = contingency_table.melt(id_vars=cat1, var_name=cat2, value_name='count')


        # Crear el heatmap
        heatmap = alt.Chart(contingency_table_long).mark_rect().encode(
            x= cat2 + ':O',
            y= cat1 + ':O',
            color=alt.Color('count:Q', scale=alt.Scale(scheme='spectral'), legend=None),
            tooltip=[cat1, cat2, 'count']
        ).properties(
            width=600,
            height=800,
            title='Heatmap de la Tabla de Contingencia'
        )


        text = alt.Chart(contingency_table_long).mark_text(baseline='middle', fontSize=20).encode(
            x=alt.X( cat2 + ':O', title=''),
            y=alt.Y( cat1 + ':O', title=''),
            text='count:Q',
            color=alt.condition(
                alt.datum['count'] > 10,
                alt.value('white'),
                alt.value('black')
            )
        )

        # Combinar el heatmap con los números
        heatmap_with_text = (heatmap + text).properties(
            width=400,
            height=600,
            title='Heatmap'
        )

        heatmap_with_text = heatmap_with_text.configure_legend(orient='none')

        st.altair_chart(heatmap_with_text, use_container_width=True)
        

def separate_df_in_X_and_y(df, target_var):
    y = np.array(df[target_var])
    X = df.drop(columns=[target_var])
    #st.write(X)
    return X, y



def preprocess_df(df, imputador_strategy, estandarizador):

    # solo a variables numericas
    num = var_classification(df)['Numerical']
    cat = var_classification(df)['Categorical']
    df_num = df[num]
    df_cat = df[cat]

    if not bool:
        st.write('')

    else:
        # Inicializa las variables
        imputador = SimpleImputer(strategy='mean')
        scaler    = StandardScaler()

        # Encoding categorical vars
        #=========================================================================
        if len(cat) != 0:
            encoder = OneHotEncoder(sparse_output=False)
            one_hot_encoded = encoder.fit_transform(df_cat)
            cols_encoded = encoder.get_feature_names_out(cat)
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=cols_encoded)
        else:
            one_hot_df = None
        #=========================================================================


        # Imputador
        if imputador_strategy == 'mean':
            imputador = SimpleImputer(strategy='mean')
        elif imputador_strategy == 'median':
            imputador = SimpleImputer(strategy='median')
        elif imputador_strategy == 'most_frequent':
            imputador = SimpleImputer(strategy='most_frequent')
        elif imputador_strategy == 'knn':
            imputador = KNNImputer()

        
        # Estandarizador
        if estandarizador == 'standard':
            scaler = StandardScaler()
        elif estandarizador == 'minmax':
            scaler = MinMaxScaler()
        
        # Aplicar el imputador
        df_imputado = pd.DataFrame(imputador.fit_transform(df_num), columns=df_num.columns)
        
        # Aplicar el estandarizador
        df_estandarizado = pd.DataFrame(scaler.fit_transform(df_imputado), columns=df_num.columns)
        
        
        # añadir las columnas one-hot-encodeadas
        df_final_encoded = pd.concat([df_estandarizado, one_hot_df], axis=1)

        st.write(f'Imputation technique    : {imputador_strategy}')
        st.write(f'Standarization technique: {estandarizador}')
        st.write(f'(One-Hot encoded columns by default)')
        st.write(df_final_encoded)

        return df_final_encoded


# Definir una función para imprimir características seleccionadas
def feature_selector(selector, X, y, target_name, type_of_model, k):

    # Eliminar posibles NaNs en la variable objetivo:

    # crear una serie de pandas con el array "y"
    y_series = pd.Series(y, name = 'target')

    # combinar el df con la variable target
    X_y = pd.concat([X, y_series], axis = 1)

    # eliminar las filas donde haya nulls en la variable target
    X_y_clean = X_y.dropna(subset=['target'])

    # separar de nuevo x e y
    X = X_y_clean.drop(columns=['target'])
    y = X_y_clean['target'].values

    #=======================================================
    

    # regresión
    if type_of_model == 'Regression':
            
        try:

            if selector != 'None':

                if selector == 'Select K Best':
                    selector_obj = SelectKBest(score_func=f_regression, k=k)

                elif selector == 'rfe':
                    model = LinearRegression()
                    selector_obj = RFE(estimator=model, n_features_to_select=k)

                elif selector == 'SelectFromModel(lasso)':
                    model = LassoCV()
                    selector_obj = SelectFromModel(estimator=model, max_features=k)
                
                elif selector == 'mrmr Regression':
                    seleccionadas = mrmr_regression(X=X, y=y, K=k, n_jobs=1)
                    X_selected = X[seleccionadas]
                    st.write(X[seleccionadas])
                    return X_selected

                selector_obj.fit(X, y)
                seleccionadas = X.columns[selector_obj.get_support()]
                X_selected = X[seleccionadas]
                #st.write(X[seleccionadas])

            else:
                st.write('No Feature Selection Applied')
                X_selected = X

            return X_selected
    
        except:
            st.write(f'Target variable {target_name} not suitable for regression')
    

    
    # clasificación
    else:
        
        try:

            if selector != 'None':

                if selector == 'Select K Best':
                    selector_obj = SelectKBest(score_func=f_classif, k=k)

                elif selector == 'rfe':
                    model = LogisticRegression(max_iter=1000)
                    selector_obj = RFE(estimator=model, n_features_to_select=k)

                elif selector == 'SelectFromModel(RF)':
                    model = RandomForestClassifier(n_estimators=100)
                    selector_obj = SelectFromModel(estimator=model, max_features=k)

                elif selector == 'mrmr Classification':
                    seleccionadas = mrmr_classif(X=X, y=y, K=k, n_jobs=1)
                    X_selected = X[seleccionadas]
                    #st.write(X[seleccionadas])
                    return X_selected
                
                selector_obj.fit(X, y)
                seleccionadas = X.columns[selector_obj.get_support()]
                X_selected = X[seleccionadas]
                #st.write(X[seleccionadas])
            
            else:
                st.write('No Feature Selection Applied')
                X_selected = X
    
            return X_selected
        
        except:
            st.write(f'Target variable {target_name} not suitable for classification')




def model_ranking(bool, X_selected_features, y, type_of_model):
        
        if bool == True:

            st.write('Loading model ranking, this might take a minute...')
            if type_of_model == 'Regression':
                X_train, X_test, y_train, y_test = train_test_split(X_selected_features,
                                                                    y,
                                                                    test_size= .33,
                                                                    random_state = 123)
                
                reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
                selected_models = ['XGBRegressor', 'Lasso', 'Ridge', 'RandomForestRegressor', 'KNeighborsRegressor']
                models, predictions = reg.fit(X_train, X_test, y_train, y_test)
                st.write(models)

            if type_of_model == 'Classification':
                X_train, X_test, y_train, y_test = train_test_split(X_selected_features,
                                                                    y,
                                                                    test_size= .33,
                                                                    random_state = 123)
                
                clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
                models,predictions = clf.fit(X_train, X_test, y_train, y_test)
                st.write(models)




        else:
            
            st.write('Load previous process')
        