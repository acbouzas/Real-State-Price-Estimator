import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import requests
from streamlit_lottie import st_lottie


def clean_outliers(df):
    # Calculate MAD
    # Select only numeric columns
    columns = df.select_dtypes(exclude=['object'])

    for column in columns:
        median = df[column].median()
        absolute_deviations = np.abs(df[column] - median)
        mad = absolute_deviations.median()

        # Define the threshold for outliers (3 MADs away from the median)
        sup_threshold = median + mad * 12
        #inf_threshold = median - mad * 12
        
        vec = df[column].copy()
         
        df[column] = np.where(vec > sup_threshold, np.nan, vec)
        
        # Replace values with NaN if they exceed the threshold        #df.loc[            df_new[column] > sup_threshold,column#        ] = np.nan
    return df

# Define a function that filters df by the user's choices

def filter_df (df, sup_up, sup_low, ant_up, ant_low,amb):
    filtered_df = df[
        (df['Superficie'] >= sup_low) & 
        (df['Superficie'] <= sup_up) & 
        (df['Antiguedad'] >= ant_low) & 
        (df['Antiguedad'] <= ant_up) &
        (df['Ambientes'] == amb)
    ]
    
    return filtered_df
    
def median_absolute_deviation(data):
    # Exclude NaN values from the data
    data = data[~np.isnan(data)]
    
    # Calculate the median
    median_value = np.nanmedian(data)
    
    # Calculate absolute deviations from the median
    abs_deviations = np.abs(data - median_value)
    
    # Calculate the MAD (median of absolute deviations)
    mad = np.nanmedian(abs_deviations)
    
    return mad


def clean_neighborhoods(df):
    
    neighborhoods_to_remove = [
    'cid campeador',
    'parque rivadavia',
    'barrio naon',
    'villa real',
    'parque las heras',
    'primera junta',
    'plaza san martin',
    'villa soldati',
    'barrio san pedro',
    'villa riachuelo',
    'catalinas',
    'capital federal',
    'barrio justo suarez',
    'parque lezica'
    ]
    
    # With drop the columns we don't need for our model
    df = df.drop(['address', 'expenses_$', 'info', 'scrape_timestamp', 'type'], axis=1)
    df = df[~df['neighborhood'].isin(neighborhoods_to_remove)]

    # Search for values containing different names for the same neighborhood and replace them with the real neighborhood 
    #df.loc[df['neighborhood'].str.contains('palermo'), 'neighborhood'] = 'palermo'
    #df.loc[df['neighborhood'].str.contains('belgrano'), 'neighborhood'] = 'belgrano'
    df.loc[df['neighborhood'].str.contains('almagro'), 'neighborhood'] = 'almagro'
    df.loc[df['neighborhood'].str.contains('flores '), 'neighborhood'] = 'flores'
    df.loc[df['neighborhood'].str.contains('river'), 'neighborhood'] = 'nunez'
    df.loc[df['neighborhood'].str.contains('nunez'), 'neighborhood'] = 'nunez'
    df.loc[df['neighborhood'].str.contains('saavedra'), 'neighborhood'] = 'saavedra'
    df.loc[df['neighborhood'].str.contains('tribunales'), 'neighborhood'] = 'san nicolas'
    df.loc[df['neighborhood'].str.contains('once'), 'neighborhood'] = 'balvanera'
    df.loc[df['neighborhood'].str.contains('abasto'), 'neighborhood'] = 'balvanera'
    df.loc[df['neighborhood'].str.contains('urquiza'), 'neighborhood'] = 'urquiza'
    df.loc[df['neighborhood'].str.contains('floresta'), 'neighborhood'] = 'floresta'
    df.loc[df['neighborhood'].str.contains('caballito'), 'neighborhood'] = 'caballito'
    df.loc[df['neighborhood'].str.contains('microcentro'), 'neighborhood'] = 'centro'
    df.loc[df['neighborhood'].str.contains('botanico'), 'neighborhood'] = 'palermo'
    df.loc[df['neighborhood'].str.contains('agronomia'), 'neighborhood'] = 'paternal'

    return df

def run_model(superficie, antiguedad, ambientes, barrio):
    # Load your trained XGBoost model (replace 'xgb_model.pkl' with your model's file path)
    model_xgb = xgb.Booster()
    model_xgb.load_model('model.json')

    with open('scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    
    # List of all neighborhood columns
    neighborhood_columns = ['almagro', 'balvanera', 'barracas', 'barrio norte', 'belgrano', 'belgrano barrancas', 'belgrano c', 'belgrano chico', 'belgrano r', 'boca', 'boedo', 'caballito', 'centro', 'chacarita', 'coghlan', 'colegiales', 'congreso', 'constitucion', 'flores', 'floresta', 'las canitas', 'liniers', 'mataderos', 'monserrat', 'monte castro', 'nunez', 'palermo', 'palermo chico', 'palermo hollywood', 'palermo nuevo', 'palermo soho', 'palermo viejo', 'parque avellaneda', 'parque centenario', 'parque chacabuco', 'parque chas', 'parque patricios', 'paternal', 'pompeya', 'puerto madero', 'recoleta', 'retiro', 'saavedra', 'san cristobal', 'san nicolas', 'san telmo', 'urquiza', 'velez sarsfield', 'versalles', 'villa crespo', 'villa del parque', 'villa devoto', 'villa general mitre', 'villa lugano', 'villa luro', 'villa ortuzar', 'villa pueyrredon', 'villa santa rita']

    # Create the dictionary with user's choices
    user_property = {
        'superficie_cubierta': superficie,
        'antiguedad': antiguedad,
        'ambientes': ambientes,
    }

    # Set all other neighborhood variables to 0
    for column in neighborhood_columns:
        if column != barrio:
            user_property[column] = 0
        if column == barrio:
            user_property[column] = 1
    # Create a DataFrame with the new observation
    new_observation = pd.DataFrame([user_property])
    new_observation_scaled = loaded_scaler.transform(new_observation)
    # Make the prediction
    predicted_price = model_xgb.predict(xgb.DMatrix(new_observation_scaled))
    price = np.exp(predicted_price[0])

    return price

def load_lottierurl (url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
def format(string):
    formatted_string = f"{int(string):,.0f}".replace(",", ".")
    return formatted_string

#Beggining of the App

data = pd.read_csv('data/departamentos.csv')

header = st.container()
selection = st.form(key='form1')

with header:
    st.markdown("<h1 style='text-align: center;'>Tasa el valor de tu Departamento</h1>", unsafe_allow_html=True)

    #st.title('Tasa el valor de tu Departamento')
    lottie_property = load_lottierurl("https://lottie.host/0b101377-5cc6-4739-b62b-deb72cb8ff63/TM9SaYTU7O.json")
    
    #right_column = st.columns(1)

    #with right_column:
    st_lottie(lottie_property, height=300, key="property")

with selection:
    st.subheader('Selecciona las caracteristicas de tu propiedad y conoce su valor al instante')
    sel_col, sel_col2 = st.columns(2)
    # Create a dynamic input for the user to select a stock symbol (ticker)
    df = clean_neighborhoods(data)
    barrios = df.neighborhood.sort_values(ascending=True).unique()
    barrio = st.selectbox(f'**Barrio**', options= barrios, index=0)
    superficie = sel_col.slider(f'**Superficie**', min_value=10, max_value=300, value=42, step=1)
    antiguedad = sel_col.slider(f'**Antiguedad**', min_value=1, max_value=100, value=40, step=4)
    ambientes = sel_col.slider(f'**Ambientes**', min_value=1, max_value=20, value=2, step=1)
    calculate_button = st.form_submit_button(label='Calcular Precio')

    if calculate_button:
        sup_up = superficie*1.1
        sup_low = superficie*0.9
        ant_up = antiguedad*1.2
        ant_low = antiguedad*0.8


        df_new = df[df.neighborhood == f"{barrio}"].copy()
        df_cleaned = clean_outliers(df_new)
        
        df_cleaned.rename(columns = {'ambientes':'Ambientes', 'superficie_cubierta':'Superficie', 'antiguedad':'Antiguedad', 'price_usd':'Precio_usd', 'neighborhood':'Barrio', 'sku_link':'Link'}, inplace=True)
        df_cleaned['Precio_mt2'] = (df_cleaned.Precio_usd/df.superficie_cubierta).round()
                # New filtered df
        filtered_df = filter_df(df_cleaned, sup_up, sup_low, ant_up, ant_low, ambientes)


        # Assuming your DataFrame is 'filtered_df' and the column name is 'Precio_mt2'

        mad = median_absolute_deviation(filtered_df['Precio_mt2'])

        if not np.isnan(mad):

            precio_mt2_median = filtered_df.Precio_mt2.median()
            precio_final = superficie*precio_mt2_median
            upper = superficie*(precio_mt2_median+mad)
            lower = superficie*(precio_mt2_median-mad)

            #st.write(f"## El precio/mt2 para tu tipo de propiedad es {precio_mt2_median}")
            st.markdown(f"#### El valor estimado esta entre USD {format(lower)} - USD {format(upper)}")
            st.markdown(f"#### El precio medio sugerido de tu propiedad es USD {format(precio_final)}")

        else: 
            st.markdown("No Hay suficientes observaciones para calcular tu precio medio.")

        price = run_model(superficie, antiguedad, ambientes, barrio)
        precio_propiedad = format(price) 
        st.write(f"#### Precio sugerido del modelo de Machine Learning es USD  {precio_propiedad}")

if calculate_button:
    lottie_coding = load_lottierurl("https://lottie.host/c2a1fe32-774c-4c98-824c-3545c265294e/L13tWK6BG6.json")
    st_lottie(lottie_coding, height=300, key="coding")

    st.write('---')
    st.write(
        """
        ### ¿Cómo se realizó el Cálculo?
        Se comparó tu tipo de propiedad con otras publicadas en sitios de búsqueda de departamentos.
        #### Precios Medios para tu Barrio
        - El cálulo tuvo en cuenta departamentos con características similares y ubicados en el mismo barrio.

        #### Precio del Modelo de Machine Learning
        - El algoritmo se entrenó con datos de departamentos de toda la capital. 
        
        Suele ser muy útil en casos en donde tu departamento está ubicado en zonas con poca oferta o tiene características particulares que hacen que no puedan encontrarse propiedades similares por la zona para realizar el *'Precio Medio'*. 


        """)





