#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


# # 1. Preparación de los datos: División en train y test

# Del conjunto de datos, nos quedamos solo con la ciudad de "Madrid". Posteriormente dividimos el dataset en conjunto de train y de test.

# In[2]:


file_path = './airbnb-listings-extract.csv'
df = pd.read_csv(file_path, delimiter=';')

df_madrid = df[df['City'] == 'Madrid']

train, test = train_test_split(df_madrid, test_size=0.2, shuffle=True, random_state=42)

print(f"Dimensiones del conjunto de training: {train.shape}")
print(f"Dimensiones del conjunto de test: {test.shape}")

train.to_csv('./train.csv', sep=';', decimal='.', index=False)
test.to_csv('./test.csv', sep=';', decimal='.', index=False)


# Ahora trabajamos solo con el conjunto de train.

# In[3]:


df = pd.read_csv('./train.csv', sep=';', decimal='.')


# # 2. Análisis exploratorio y Preprocesamiento

# He decidido mezclar ambos apartados porque del análisis exploratorio voy a llegar a determinadas conclusiones para ir eliminando variables irrelevantes, o con outliers o con datos nulos mientras avanzo.

# ## 2.1 Head, info, describe: Una visual a nuestros datos

# In[4]:


df.head() # Una primera visual de las primeras 5 filas de mi dataset


# In[5]:


df.info() # Nos da información sobre todas las variables de mi modelo, si contiene nulos y el tipo


# In[6]:


df.describe() # Nos ofrece una descripción estadíistica del dataset por cada variable


# ## 2.2 Eliminación de algunas variables  

# En df.info() hemos obtenido mucha información de nuestras variables, y nos ha dado una primera impresión de variables que se pueden eliminar de nuestro modelo porque no aportan información valiosa para nuestro análisis.

# Estas variables son: 
# - Identificadores y URLs: ID, Listing Url, Scrape ID, Host ID ,Host URL, Thumbnail Url, Medium Url, Picture Url, XL Picture Url, Host Thumbnail Url, Host Picture Url
# - Fechas de actualizaciones: Last Scraped, Calendar Updated, Calendar last Scraped
# - Textos descriptivos: Name, Summary, Space, Description, Neighborhood Overview, Notes, Transit, Access, Interaction, House Rules, Host Name, Host About, Host Neighbourhood, Host Location, Host Verifications, First Review, Last Review
# - Variables con muchos nulos que dejan de ser relevantes y no tenemos suficientes datos para imputar: Host Acceptance Rate, Jurisdiction Names, Has Availability, License, Square Feet
# - Variables redundantes: Availability 30, Availability 60, Availability 90 (dejamos Availability 365), Weekly Price, Monthly Price (para que no afecten a Price que es nuestra dependiente)
# - Variables de ubicación: Street, Geolocation, Smart Location (ya tenemos latitud y longitud), Country (solo vemos Madrid), Zipcode (ya tenemos Neighbourhood), Market, State

# In[7]:


columns_to_drop = [
    'ID', 'Listing Url', 'Scrape ID', 'Host ID', 'Host URL', 'Thumbnail Url', 'Medium Url', 
    'Picture Url', 'XL Picture Url', 'Host Thumbnail Url', 'Host Picture Url', 
    'Last Scraped', 'Calendar Updated', 'Calendar last Scraped', 
    'Name', 'Summary', 'Space', 'Description', 'Neighborhood Overview', 'Notes', 
    'Transit', 'Access', 'Interaction', 'House Rules', 'Host Name', 'Host About', 
    'Host Neighbourhood', 'Host Location', 'Host Verifications', 'First Review', 'Last Review',
    'Host Acceptance Rate', 'Jurisdiction Names', 'Has Availability', 
    'License', 'Square Feet', 'Availability 30', 'Availability 60', 'Availability 90', 
    'Weekly Price', 'Monthly Price', 'Street', 'Geolocation', 'Smart Location', 
    'Country', 'Zipcode', 'Market', 'State'
]

df.drop(columns=columns_to_drop, axis=1, inplace=True)

df.info()


# Nos quedamos de momento con estas variables, pero también considero que las variables Review pueden estar correlacionadas al igual que las variables de "Listing Count", pero prefiero verlo en el siguiente apartado y tratarlas de forma especial.

# ## 2.3 Correlación 

# ### Aquí también eliminaremos otras variables y generaremos nuevas a partir de otras variables. 

# In[8]:


corr = np.abs(df.drop(['Price'], axis=1).corr())

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

plt.show()


# Un umbral común para detectar la multicolinealidad es 0.8 (en valor absoluto), de acuerdo con autores como:
# > Kennedy (2008)
# 
# > Pal and Soriya (2012)

# Por ello, voy a establecer este umbral para determinar su presencia en mis datos.

# In[9]:


threshold = 0.8
high_corr_var = [(corr.columns[i], corr.columns[j])
                 for i in range(len(corr.columns))
                 for j in range(i+1, len(corr.columns))
                 if corr.iloc[i, j] > threshold]

print("Pares de variables con correlación superior a 0.8:")
for var1, var2 in high_corr_var:
    print(f"{var1} - {var2}: {corr.loc[var1, var2]}")


# Las conclusiones que saco son las siguientes:
# - Entre Accomodates y Beds, podemos eliminar directamente Accomodates.
# - Entre las variables Review hay cierta correlación, pero solo voy a tratar el caso más grave que es entre Review Scores Rating y Review Scores Value, donde voy a hacer una nueva variable con la media de ambas.
# - Como veíamos antes, las variables Host Listings Count, Host Total Listings Count y Calculated host listings count, tienen una correlación prácticamente directa, por lo que se puede hacer también una nueva variable con la media de las tres. Considero que esta variable es importante porque el número de propiedades de un anfitrión puede afectar de forma escalada al precio al que pone a sus alojamientos pudiendo llegar a ser incluso más competitivos que un anfitrión que solo tenga una sola propiedad.

# In[10]:


df['Host Listings Combined'] = df[['Calculated host listings count', 'Host Total Listings Count', 'Host Listings Count']].mean(axis=1)
df['Review Combined'] = df[['Review Scores Rating', 'Review Scores Value']].mean(axis=1)
df.drop(columns=['Accommodates','Calculated host listings count', 'Host Total Listings Count', 'Host Listings Count', 'Review Scores Rating', 'Review Scores Value'], axis=1, inplace=True)


# In[11]:


corr = np.abs(df.drop(['Price'], axis=1).corr())

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

plt.show()


# In[12]:


df.head() # Me he quedado con 39 columnas


# Comprobamos de nuevo nuestra matriz de correlación para ver que todo va mejorando. Ahora haremos un scatter plot para ver cómo se relacionan todas las variables entre sí.

# In[13]:


#pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(20, 20), diagonal = 'kde')
#plt.show()


# ## 2.4 Outliers

# Voy a definir una función que identifique los valores atípicos o outliers en variables numéricas usando el rango intercuartílico (IQR) para cada una de ellas. De este modo, veremos de un vistazo, posibles variables a tener en cuenta para eliminar outliers.

# In[14]:


# Veo el número de variables numéricas que tengo

num_float64 = df.select_dtypes(include=['float64']).shape[1]
num_int64 = df.select_dtypes(include=['int64']).shape[1]

print(f"Número de columnas tipo float64: {num_float64}")
print(f"Número de columnas tipo int64: {num_int64}")
print(f"Un total de {num_float64+num_int64} variables numéricas")


# In[15]:


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Hago la función para IQR

def find_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    return outliers

outliers_summary = {}
for col in numeric_cols:
    outliers = find_outliers_iqr(df[col])
    outliers_summary[col] = outliers.sum()

outliers_summary = pd.DataFrame.from_dict(outliers_summary, orient='index', columns=['Número de outliers según IQR'])
outliers_summary = outliers_summary.sort_values(by='Número de outliers según IQR', ascending=False)
print(outliers_summary)


# Revisando las variables con posibles outliers, voy a hacer una serie de gráficos de las 5 primeras que más outliers tienen. Pero voy a dejar fuera a Longitude, que es más amplia, y a meter Price, puesto que es mi dependiente y me conviene quitar los outliers. En concreto, haré un histograma, un bloxplot de cada una y un scatter plot con la variable Price. De este modo podré visualizar cuál o cuáles tiene más sentido quitar.

# In[16]:


outliers_variables = ['Bathrooms','Host Response Rate','Host Listings Combined','Beds','Price']

for k in outliers_variables:
    plt.figure(figsize=(15, 6))
    
    # Histograma
    plt.subplot(1, 3, 1)
    sns.histplot(df[k], bins=30, kde=True)
    plt.title('Histograma de ' + k)
    plt.xlabel(k)
    plt.ylabel('Frecuencia')

    # Boxplot
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df[k])
    plt.title('Boxplot de ' + k)

    # Scatter plot con Price
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=df[k], y=df['Price'])
    plt.title(k + ' vs Price')
    plt.xlabel(k)
    plt.ylabel('Price')

    plt.tight_layout()
    plt.show()


# LIMPIEZA --> Antes de seguir con los outliers, he visto que la variable Bathrooms tiene valores medios, como 1.5, 2.5, etc. Creo que esto no se debe dar, por lo que voy a proceder a quitar esos datos, ya que los considero erróneos.

# In[17]:


df['Bathrooms'].value_counts()


# In[18]:


df = df[~df['Bathrooms'].isin([0.5,1.5, 2.5,3.5,4.5,5.5,6.5])]


# In[19]:


df.shape


# Siguiendo con los outliers, creo que debemos eliminar outliers de las siguientes variables para quedarnos con los datos que considero más representativos:
# - Bathrooms: No tiene sentido propiedades con 0 baños. Haré el corte en menos de 5 baños, inclusive.
# - Host Response Rate: Haré el corte en los que tienen más de 60, inclusive.
# - Host Listings Combined: Haré el corte en menos de 75 propiedades, no inluído.
# - Beds: Haré el corte en menos de 10 camas, inclusive.
# - Price: Haré el corte en un mínimo de 400 euros, inclusive.
# 
# No obstante, voy a ver el análisis de value_counts de cada una de estas variables para ver si modifico algo.
# 

# In[20]:


for k in outliers_variables:
    print(df[k].value_counts())


# In[21]:


# Filtramos el dataframe sin los outliers haciendo uno nuevo.

conditions = (
    (df['Bathrooms'] > 0) &
    (df['Bathrooms'] <= 5) &
    (df['Host Response Rate'] >= 60) &
    (df['Host Listings Combined'] < 75) &
    (df['Beds'] <= 10) &
    (df['Price'] <= 400))

df_no_outliers = df[conditions]

print(f"Tamaño del dataframe original: {df.shape}")
print(f"Tamaño del dataframe sin outliers: {df_no_outliers.shape}")


# In[22]:


print(
    f'Original: {df.shape[0]} // '
    f'Modificado: {df_no_outliers.shape[0]}\nDiferencia: {df.shape[0] - df_no_outliers.shape[0]}'
)
print(f'Variación: {(((df.shape[0] - df_no_outliers.shape[0])/df.shape[0])*100):.2f}%')


# La verdad es que de este modo me quedo con pocos datos, es una bajada muy importante, por lo que le voy a dar otro enfoque. Me centraré solo en Bathrooms (entre 0 y 5), Beds (menos de 10) y Price (menos de 400), que las considero más relevantes.

# In[23]:


# Hacemos de nuevo el dataframe sin los outliers.

conditions = (
    (df['Bathrooms'] > 0) &
    (df['Bathrooms'] <= 5) &
    (df['Beds'] <= 10) &
    (df['Price'] <= 400))


df_no_outliers = df[conditions]

print(f"Tamaño del dataframe original: {df.shape}")
print(f"Tamaño del dataframe sin outliers: {df_no_outliers.shape}")


# In[24]:


print(
    f'Original: {df.shape[0]} // '
    f'Modificado: {df_no_outliers.shape[0]}\nDiferencia: {df.shape[0] - df_no_outliers.shape[0]}'
)
print(f'Variación: {(((df.shape[0] - df_no_outliers.shape[0])/df.shape[0])*100):.2f}%')


# Ahora sí hemos conseguido una variación modesta y con un dataset sin los principales outliers.

# In[25]:


df_no_outliers.shape


# In[26]:


df_no_outliers.describe()


# In[27]:


# Voy a ver de nuevo la matriz de correlación para comprobar que todo sigue teniendo sentido

corr = np.abs(df_no_outliers.drop(['Price'], axis=1).corr())

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

plt.show()


# ## 2.5 Categorización de variables y generación de nuevas características

# ### Aquí también terminaremos de limpiar el dataset.

# Lo primero que haré es sacar el listado de las columnas no numércias, que son éstas.

# In[28]:


categoricas = df_no_outliers[list(df_no_outliers.select_dtypes(include=['object']).columns)]
categoricas.head()


# Voy a comprobar los valores únicos de cada una.

# In[29]:


categoricas.apply(lambda x: len(x.unique()))


# Primero me centraré en la categorización de variables, por lo que voy a fijarme en aquellas que tengan pocos valores únicos y a revisar su forma para ver si son ordinales (Label Encoder), ordinales (One-Hot-Encoder) o si hay una alta cardinalidad (Target Encoder). 
# 
# En este sentido, me centraré primero en:
# - Experiences Offered
# - City
# - Country Code
# - Host Response Time
# - Property Type
# - Room Type
# - Bed Type 
# - Cancellation Policy
# 
# El resto de variables que tienen más registros únicos, las consideraré para la generación de nuevas características.

# Vemos que Experience Offered solo tiene un valor "None", por lo que podemos eliminarla directamente al no aportar información relevante. Lo mismo pasa con City, que solo es Madrid, y con Country Code. Procedemos a eliminarlas directamente.

# In[30]:


df_no_outliers.drop(columns=['Experiences Offered', 'City', 'Country Code'], axis=1, inplace=True)


# Voy a revisar el resto de variables con value_counts para hacerme una idea de su forma.

# In[31]:


df_no_outliers['Host Response Time'].value_counts() # Para ésta usaremos un Label Encoding al ser una variable ordinall


# In[32]:


response_time_order = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']
df_no_outliers['Host Response Time'] = pd.Categorical(df_no_outliers['Host Response Time'], categories=response_time_order, ordered=True)
le_host_response_time = LabelEncoder()
df_no_outliers['Host Response Time'] = le_host_response_time.fit_transform(df_no_outliers['Host Response Time'])


# In[33]:


category_counts_property = df_no_outliers['Property Type'].value_counts()
category_counts_property


# Hacer One-Hot-Encoding con esta variable añadiría demasiado ruido a mi dataset, y no es lo que queremos, ya que nos generaría 20 variables nuevas. He decidido reducir la dimensionalidad quedándome solo con registros que sean una muestra representativa. He probado con varios parámetros y me voy a quedar con los que sean superiores a 100 registros.

# In[34]:


threshold = 100

frequent_categories_property = category_counts_property[category_counts_property > threshold].index.tolist()

df_no_outliers = df_no_outliers[df_no_outliers['Property Type'].isin(frequent_categories_property)]


# In[35]:


df_no_outliers['Property Type'].value_counts()


# In[36]:


df_no_outliers = pd.get_dummies(df_no_outliers, columns=['Property Type'], prefix='PropertyType')
df_no_outliers.head()


# In[37]:


df_no_outliers['Room Type'].value_counts()


# In[38]:


df_no_outliers = pd.get_dummies(df_no_outliers, columns=['Room Type'], prefix='RoomType') # One-Hot-Encoding
df_no_outliers.head()


# In[39]:


category_counts_bed = df_no_outliers['Bed Type'].value_counts()
category_counts_bed


# Considero igual que en Property Type; no tiene sentido quedarme con tan pocos registros, que no son representativos, por lo que voy a quedarme con los varlores por encima de 100.

# In[40]:


threshold = 100

frequent_categories_bed = category_counts_bed[category_counts_bed > threshold].index.tolist()

df_no_outliers = df_no_outliers[df_no_outliers['Bed Type'].isin(frequent_categories_bed)]


# In[41]:


df_no_outliers = pd.get_dummies(df_no_outliers, columns=['Bed Type'], prefix='BedType')
df_no_outliers.head()


# In[42]:


category_counts_cp = df_no_outliers['Cancellation Policy'].value_counts()
category_counts_cp


# Voy a eliminar las filas que contienen solo dos registros. Pondré el mismo nivel, me quedo con lo que esté superior a 100.

# In[43]:


threshold = 100

frequent_categories_cp = category_counts_cp[category_counts_cp > threshold].index.tolist()

df_no_outliers = df_no_outliers[df_no_outliers['Cancellation Policy'].isin(frequent_categories_cp)]


# In[44]:


df_no_outliers = pd.get_dummies(df_no_outliers, columns=['Cancellation Policy'], prefix='CancPolicy')
df_no_outliers.head()


# Ahora me centraré en el resto de variables para ver cómo puedo tratarlas:
# - Neighbourhood, Neighbourhood Cleansed, Neighbourhood Group Cleansed 
# - Host Since
# - Amenities
# - Features

# In[45]:


# Primero voy a ver las 3 variables de Neighbouhood a ver si comparten características

df_neighbourhood = df_no_outliers[['Neighbourhood','Neighbourhood Cleansed', 'Neighbourhood Group Cleansed']]
df_neighbourhood.head(15)


# In[46]:


len(df_neighbourhood) - df_neighbourhood.count() # Comprobamos los valores NaN de cada columna


# Viendo estos valores, podemos combinar las tres variables para quedarnos con una sola combinada.

# In[47]:


df_no_outliers['Combined_Neighbourhood'] = df_no_outliers['Neighbourhood'].combine_first(df_no_outliers['Neighbourhood Cleansed']).combine_first(df_no_outliers['Neighbourhood Group Cleansed'])
df_no_outliers.drop(columns=['Neighbourhood', 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed'], axis=1, inplace=True)

df_no_outliers.head()


# Al tener múltiples categorías, haré un Target Encoding con la media del precio (Price), que es nuestra variable dependiente, para la nueva columna combinada y eliminaré dicha columna.

# In[48]:


mean_encoded = df_no_outliers.groupby('Combined_Neighbourhood')['Price'].mean()

df_no_outliers['Combined_Neighbourhood_Encoded'] = df_no_outliers['Combined_Neighbourhood'].map(mean_encoded)
df_no_outliers.drop(columns=['Combined_Neighbourhood'], axis=1, inplace=True)
df_no_outliers.head()


# Con la antigüedad del dueño podemos generar nuevas características a partir de Host Since, como el año, el mes y los días que lleva el anfitrión en la plataforma.

# In[49]:


df_no_outliers['Host Since'].value_counts() 


# In[50]:


df_no_outliers['Host Since'] = pd.to_datetime(df_no_outliers['Host Since'])

df_no_outliers['Host Since Year'] = df_no_outliers['Host Since'].dt.year
df_no_outliers['Host Since Month'] = df_no_outliers['Host Since'].dt.month

today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
df_no_outliers['Days Since Host'] = (today - df_no_outliers['Host Since']).dt.days
df_no_outliers.drop(columns=['Host Since'], axis=1, inplace=True)


# In[51]:


df_no_outliers.head()


# In[52]:


df_no_outliers['Amenities'].value_counts() 


# Esta variable tiene múltiples comidades dentro de los alojamientos, pero están separados por comas. Podemos dividirlos por esas características y hacer un recuento de cada una de ellas en cada fila.

# In[53]:


df_no_outliers['Amenities'] = df_no_outliers['Amenities'].fillna('') # Rellenamos con huecos vacíos los NaN, significa que no hay ninguna comodidad en el alojamiento
df_no_outliers['Amenities'] = df_no_outliers['Amenities'].apply(lambda x: x.split(','))
df_no_outliers['Amenities'] = df_no_outliers['Amenities'].apply(len) 
df_no_outliers.head()


# In[54]:


df_no_outliers['Features'].value_counts() 


# Esta última variable Features la voy a tratar del mismo modo que la anterior.

# In[55]:


df_no_outliers['Features'] = df_no_outliers['Features'].fillna('') 
df_no_outliers['Features'] = df_no_outliers['Features'].apply(lambda x: x.split(','))
df_no_outliers['Features'] = df_no_outliers['Features'].apply(len)
df_no_outliers.head()


# ### Termino de limpiar el dataset rellenando los datos NaN.

# In[56]:


# Contamos los datos NaN de cada variable

nan_counts = df_no_outliers.isnull().sum()
nan_counts


# Lo primero que veo es que puedo eliminar directamente los que vienen de Host Since y Bedrooms, porque solo son dos y 9 filas, respectivamente.

# In[57]:


df_no_outliers.dropna(subset=['Host Since Year','Host Since Month', 'Days Since Host', 'Bedrooms'], inplace=True)


# Ahora voy a trabajar con Host Response Rate, Security Deposit, y Cleaning Fee. Voy a ver sus distribuciones para ver cuál es la mejor forma de rellenar sus NaN.

# In[58]:


def plot_distribution(df, column_name):
    plt.figure(figsize=(12, 6))

    # Histograma
    plt.subplot(1, 2, 1)
    sns.histplot(df[column_name], kde=True, bins=30)
    plt.title(f'Histograma de {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.show()

plot_distribution(df_no_outliers, 'Host Response Rate')
plot_distribution(df_no_outliers, 'Security Deposit')
plot_distribution(df_no_outliers, 'Cleaning Fee')


# No tienen distribuciones normales, por lo que, rellenar con su media puede no ser la mejor opción. Voy a optar por rellenarlos con la mediana, puesto que es menos sensible a valores atípicos.

# In[59]:


def fill_na_with_median(df, column_name):
    median_value = df[column_name].median()
    df[column_name].fillna(median_value, inplace=True)
    
    return df

columns_to_fill = ['Host Response Rate', 'Security Deposit', 'Cleaning Fee']
for column in columns_to_fill:
    df_no_outliers = fill_na_with_median(df_no_outliers, column)
    
df_no_outliers.head()


# Finalmente, veré las variables Review para terminar de rellenar sus datos.
# Dado que estas variables tienen relación entre sí, he pensado buscar otro método más sofisticado para rellenar sus NaN y preservar esta relación entre ellas.

# En concreto, voy a usar el método MICE (Multiple Imputation by Chained Equations) para rellenar estas variables con datos faltantes. MICE utiliza un enfoque iterativo y probabilístico para generar múltiples valores imputados para cada dato faltante, considerando las relaciones entre las variables, por lo que creo que puede ser ideal para este conjunto de Reviews.
# Funciona de la siguiente manera:
# 
# Para cada variable con valores faltantes, MICE ajusta un modelo de predicción utilizando las otras variables como predictores y utiliza el modelo ajustado para imputar los valores faltantes en la variable actual.
# Este proceso se repite para cada variable con valores faltantes en el conjunto de datos.
# Se realizan múltiples iteraciones (cadenas) hasta que las imputaciones convergen y se estabilizan, proporcionando estimaciones más precisas.

# In[60]:


columns_to_fill_with_mice = ['Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin',
                             'Review Scores Communication', 'Review Scores Location', 'Reviews per Month', 'Review Combined']

imputer = IterativeImputer(random_state=42)
df_mice_part = df_no_outliers[columns_to_fill_with_mice]
df_mice_imputed = pd.DataFrame(imputer.fit_transform(df_mice_part), columns=df_mice_part.columns, index=df_mice_part.index)

df_no_outliers.update(df_mice_imputed)
df_no_outliers.head()


# In[61]:


# Compruebo que todas las variables son numéricas y que verdaderamente no contienen nulos para avanzar

df_no_outliers.info()


# In[62]:


# También voy a repetir el describe para volver a hacer una visual de mis datos y que no haya nada "raro"

df_no_outliers.describe()


# Genial. Ya podemos pasar a la parte de modelado, donde tomaremos Price como variable dependiente. Para ello, la pasaremos a la primera columna de nuestro dataset.

# In[63]:


df_no_outliers = df_no_outliers[['Price'] + [col for col in df_no_outliers.columns if col != 'Price']]
df_no_outliers.head()


# # 3. Modelado

# ## 3.1 Preparamos los datos de train y test para el modelado

# In[64]:


# Preparamos los datos de train

data = df_no_outliers.values

y_train = data[:,0:1] # Price como dependiente
X_train = data[:,1:]      

feature_names = df_no_outliers.columns[1:]

scaler = preprocessing.StandardScaler().fit(X_train) # Escalamos los datos
X_train_scaled = scaler.transform(X_train)


# In[65]:


# Ponemos todo el preprocesamiento en una sola celda para ejecutarla en test

df_test = pd.read_csv('./test.csv', sep=';', decimal='.')
columns_to_drop = [
    'ID', 'Listing Url', 'Scrape ID', 'Host ID', 'Host URL', 'Thumbnail Url', 'Medium Url', 
    'Picture Url', 'XL Picture Url', 'Host Thumbnail Url', 'Host Picture Url', 
    'Last Scraped', 'Calendar Updated', 'Calendar last Scraped', 
    'Name', 'Summary', 'Space', 'Description', 'Neighborhood Overview', 'Notes', 
    'Transit', 'Access', 'Interaction', 'House Rules', 'Host Name', 'Host About', 
    'Host Neighbourhood', 'Host Location', 'Host Verifications', 'First Review', 'Last Review',
    'Host Acceptance Rate', 'Jurisdiction Names', 'Has Availability', 
    'License', 'Square Feet', 'Availability 30', 'Availability 60', 'Availability 90', 
    'Weekly Price', 'Monthly Price', 'Street', 'Geolocation', 'Smart Location', 
    'Country', 'Zipcode', 'Market', 'State'
]

df_test.drop(columns=columns_to_drop, axis=1, inplace=True)

df_test['Host Listings Combined'] = df_test[['Calculated host listings count', 'Host Total Listings Count', 'Host Listings Count']].mean(axis=1)

df_test['Review Combined'] = df_test[['Review Scores Rating', 'Review Scores Value']].mean(axis=1)

df_test.drop(columns=['Accommodates','Calculated host listings count', 'Host Total Listings Count', 'Host Listings Count', 'Review Scores Rating', 'Review Scores Value'], axis=1, inplace=True)

df_test = df_test[~df_test['Bathrooms'].isin([0.5,1.5, 2.5,3.5,4.5,5.5,6.5])]

conditions = (
    (df_test['Bathrooms'] > 0) &
    (df_test['Bathrooms'] <= 5) &
    (df_test['Beds'] <= 10) &
    (df_test['Price'] <= 400))

df_test = df_test[conditions]

df_test.drop(columns=['Experiences Offered', 'City', 'Country Code'], axis=1, inplace=True)

df_test['Host Response Time'] = pd.Categorical(df_test['Host Response Time'], categories=response_time_order, ordered=True)
df_test['Host Response Time'] = le_host_response_time.fit_transform(df_test['Host Response Time'])

df_test = df_test[df_test['Property Type'].isin(frequent_categories_property)]
df_test = pd.get_dummies(df_test, columns=['Property Type'], prefix='PropertyType')

df_test = pd.get_dummies(df_test, columns=['Room Type'], prefix='RoomType')

df_test = df_test[df_test['Bed Type'].isin(frequent_categories_bed)]
df_test = pd.get_dummies(df_test, columns=['Bed Type'], prefix='BedType')

df_test = df_test[df_test['Cancellation Policy'].isin(frequent_categories_cp)]
df_test = pd.get_dummies(df_test, columns=['Cancellation Policy'], prefix='CancPolicy')

df_test['Combined_Neighbourhood'] = df_test['Neighbourhood'].combine_first(df_test['Neighbourhood Cleansed']).combine_first(df_test['Neighbourhood Group Cleansed'])
df_test.drop(columns=['Neighbourhood', 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed'], axis=1, inplace=True)
df_test['Combined_Neighbourhood_Encoded'] = df_test['Combined_Neighbourhood'].map(mean_encoded)
df_test.drop(columns=['Combined_Neighbourhood'], axis=1, inplace=True)

df_test['Host Since'] = pd.to_datetime(df_test['Host Since'])
df_test['Host Since Year'] = df_test['Host Since'].dt.year
df_test['Host Since Month'] = df_test['Host Since'].dt.month
df_test['Days Since Host'] = (today - df_test['Host Since']).dt.days
df_test.drop(columns=['Host Since'], axis=1, inplace=True)

df_test['Amenities'] = df_test['Amenities'].fillna('')
df_test['Amenities'] = df_test['Amenities'].apply(lambda x: x.split(','))
df_test['Amenities'] = df_test['Amenities'].apply(len) 

df_test['Features'] = df_test['Features'].fillna('') 
df_test['Features'] = df_test['Features'].apply(lambda x: x.split(','))
df_test['Features'] = df_test['Features'].apply(len)

df_test.dropna(subset=['Host Since Year','Host Since Month', 'Days Since Host', 'Bedrooms'], inplace=True)

for column in columns_to_fill:
    df_test = fill_na_with_median(df_test, column)

df_mice_part = df_test[columns_to_fill_with_mice]
df_mice_imputed = pd.DataFrame(imputer.fit_transform(df_mice_part), columns=df_mice_part.columns, index=df_mice_part.index)

df_test.update(df_mice_imputed)

df_test = df_test[['Price'] + [col for col in df_test.columns if col != 'Price']]


# In[66]:


df_test.shape


# In[67]:


df_test.head()


# In[68]:


df_test.info()


# In[69]:


df_test.isnull().sum()


# Vemos que tenemos dos valores faltantes en Combined_Neighbourhood_Encoded, lo voy a rellenar con la moda de nuestro dataset de train.

# In[70]:


df_test['Combined_Neighbourhood_Encoded'].fillna(df_no_outliers['Combined_Neighbourhood_Encoded'].mode()[0], inplace=True)
df_test.isnull().sum()


# In[71]:


# Ahora sí podemos preparar los datos de test

data_test = df_test.values

y_test = data_test[:,0:1] # Primera columna Price como dependiente
X_test = data_test[:,1:]

feature_names_test = df_test.columns[1:]

X_test_scaled = scaler.transform(X_test) # Aplicamos el escalado con el scaler anterior de train


# ## 3.2 Evaluación de modelos con validación cruzada y selección de características: Lasso, Random Forest y XGBoost

# Defino las funciones que voy a usar para la evaluación y muestra de resultados en general.

# In[72]:


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f"{model_name} - MSE (Train): {mse_train}, MSE (Test): {mse_test}, RMSE (Train): {rmse_train}, RMSE (Test): {rmse_test}, R2 (Train): {r2_train}, R2 (Test): {r2_test}")

    # Gráfico de los valores reales vs los valores predichos
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.title(f'{model_name} - Reales vs Predichos')
    plt.show()

    # Gráfico para ver el error MSE a lo largo de las iteraciones (para Random Forest)
    if hasattr(model, 'estimators_'):
        mse_train_list = []
        mse_test_list = []
        for i, estimator in enumerate(model.estimators_):
            y_train_pred_iter = estimator.predict(X_train)
            y_test_pred_iter = estimator.predict(X_test)
            mse_train_iter = mean_squared_error(y_train, y_train_pred_iter)
            mse_test_iter = mean_squared_error(y_test, y_test_pred_iter)
            mse_train_list.append(mse_train_iter)
            mse_test_list.append(mse_test_iter)
        
        plt.figure(figsize=(10, 6))
        plt.plot(mse_train_list, label='MSE Train')
        plt.plot(mse_test_list, label='MSE Test')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title(f'{model_name} - MSE Over Iterations')
        plt.legend()
        plt.show()

    return [model_name, mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test]

def print_results_table(results, title):
    df_results = pd.DataFrame(results, columns=['Model','MSE Train', 'MSE Test', 'RMSE Train', 'RMSE Test', 'R2 Train', 'R2 Test'])
    df_results.set_index('Model', inplace=True)
    df_results_styled = df_results.style.set_caption(title)
    display(df_results_styled)


# ### 3.2.1 Lasso: Validación cruzada y selección de características

# In[73]:


# Validación cruzada 
alpha_vector = np.logspace(-4, 0, 50)
param_grid = {'alpha': alpha_vector}
grid_lasso = GridSearchCV(Lasso(), param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2)
grid_lasso.fit(X_train_scaled, y_train)

# Mejor alpha encontrado
best_alpha = grid_lasso.best_params_['alpha']
best_lasso = Lasso(alpha=best_alpha)

best_lasso.fit(X_train_scaled, y_train)

coefs = pd.Series(best_lasso.coef_, index=feature_names)
selected_features_lasso = coefs[coefs != 0].index.tolist()

print("Características seleccionadas por Lasso:")
print(selected_features_lasso)

X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

# Filtramos las características seleccionadas en el conjunto de entrenamiento y prueba
X_train_selected_lasso = X_train.loc[:, selected_features_lasso]
X_test_selected_lasso = X_test.loc[:, selected_features_lasso]

X_train_imputed_selected = imputer.fit_transform(X_train_selected_lasso)
X_test_imputed_selected = imputer.transform(X_test_selected_lasso)
X_train_scaled_lasso = scaler.fit_transform(X_train_imputed_selected)
X_test_scaled_lasso = scaler.transform(X_test_imputed_selected)

# Evaluación con Lasso
results_lasso = []
results_lasso.append(evaluate_model(best_lasso, X_train_scaled_lasso, y_train, X_test_scaled_lasso, y_test, 'Lasso Regression'))
print_results_table(results_lasso, "Resultados de Lasso")


# ### 3.2.2 Random Forest: Validación cruzada y selección de características

# In[74]:


from sklearn.feature_selection import SelectFromModel

# Dejo estos parámetros tras varias pruebas de estimadores, profundidad...

param_grid_rf = {
    'n_estimators': [10, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20],
}

# Vaidación cruzada
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=10, scoring='neg_mean_squared_error', verbose=2)
grid_rf.fit(X_train_scaled, y_train)

# Mejor modelo 
best_rf = grid_rf.best_estimator_

# Seleccionamos las características importantes
selector_rf = SelectFromModel(best_rf, threshold='median', prefit=True)
X_train_rf_selected = selector_rf.transform(X_train_scaled)
X_test_rf_selected = selector_rf.transform(X_test_scaled)

selected_features_rf = feature_names[selector_rf.get_support()]

print("Características seleccionadas por Random Forest:")
print(selected_features_rf)

# Evaluación con Random Forest
results_rf = []
results_rf.append(evaluate_model(best_rf, X_train_rf_selected, y_train, X_test_rf_selected, y_test, 'Random Forest'))
print_results_table(results_rf, "Resultados de Random Forest")

# Importancia
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Importancia de las características - Random Forest")
plt.bar(range(X_train_rf_selected.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_rf_selected.shape[1]), selected_features_rf[indices], rotation=90)
plt.xlim([-1, X_train_rf_selected.shape[1]])
plt.show()


# ### 3.2.3 XGBoost: Validación cruzada y selección de características

# In[75]:


# Validación cruzada para encontrar los mejores hiperparámetros para XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
}
grid_xgb = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid_xgb, cv=10, scoring='neg_mean_squared_error', verbose=2)
grid_xgb.fit(X_train_scaled, y_train)

# Mejor modelo encontrado
best_xgb = grid_xgb.best_estimator_

# Seleccionar características importantes
selector_xgb = SelectFromModel(best_xgb, threshold='median', prefit=True)
X_train_xgb_selected = selector_xgb.transform(X_train_scaled)
X_test_xgb_selected = selector_xgb.transform(X_test_scaled)

# Obtener los nombres de las características seleccionadas
selected_features_xgb = feature_names[selector_xgb.get_support()]

print("Características seleccionadas por XGBoost:")
print(selected_features_xgb)

# Evaluación con XGBoost
results_xgb = []
results_xgb.append(evaluate_model(best_xgb, X_train_xgb_selected, y_train, X_test_xgb_selected, y_test, 'XGBoost'))
print_results_table(results_xgb, "Resultados de XGBoost")

# Importancia de características
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Importancia de las características - XGBoost")
plt.bar(range(X_train_xgb_selected.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_xgb_selected.shape[1]), selected_features_xgb[indices], rotation=90)
plt.xlim([-1, X_train_xgb_selected.shape[1]])
plt.show()


# In[76]:


# Muestra de todos los resultados con selección de características

all_results = results_lasso + results_rf + results_xgb
print_results_table(all_results, "Comparación de los modelos con selección de características")


# ## 3.3 Evaluación de modelos con validación cruzada y todas las características: Lasso, Random Forest y XGBoost

# ### 3.3.1 Lasso: Validación cruzada y todas las características

# In[77]:


# Validación cruzada 
alpha_vector = np.logspace(-4, 0, 50)
param_grid = {'alpha': alpha_vector}
grid_lasso = GridSearchCV(Lasso(), param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=2)
grid_lasso.fit(X_train_scaled, y_train)

# Mejor alpha encontrado
best_alpha = grid_lasso.best_params_['alpha']
best_lasso = Lasso(alpha=best_alpha)

results_lasso_todo = []
results_lasso_todo.append(evaluate_model(best_lasso, X_train_scaled, y_train, X_test_scaled, y_test, 'Lasso Regression'))
print_results_table(results_lasso_todo, "Resultados de Lasso")


# ### 3.3.2 Random Forest: Validación cruzada y todas las características

# In[78]:


# Dejo estos parámetros tras varias pruebas de estimadores, profundidad...

param_grid_rf = {
    'n_estimators': [10, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20],
}

# Validación cruzada
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=10, scoring='neg_mean_squared_error', verbose=2)
grid_rf.fit(X_train_scaled, y_train)

# Mejor modelo 
best_rf = grid_rf.best_estimator_

results_rf_todo = []
results_rf_todo.append(evaluate_model(best_rf, X_train_scaled, y_train, X_test_scaled, y_test, 'Random Forest'))
print_results_table(results_rf_todo, "Resultados de Random Forest")

# Importancia 
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Importancia de las características - Random Forest")
plt.bar(range(X_train_scaled.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train_scaled.shape[1]])
plt.show()


# ### 3.3.3 XGBoost: Validación cruzada y todas las características

# In[79]:


# Dejo estos parámetros tras varias pruebas de estimadores, profundidad...

param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
}

# Validación cruzada
grid_xgb = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid_xgb, cv=10, scoring='neg_mean_squared_error', verbose=2)
grid_xgb.fit(X_train_scaled, y_train)

# Mejor modelo 
best_xgb = grid_xgb.best_estimator_

results_xgb_todo = []
results_xgb_todo.append(evaluate_model(best_xgb, X_train_scaled, y_train, X_test_scaled, y_test, 'XGBoost'))
print_results_table(results_xgb_todo, "Resultados de XGBoost")

# Importancia
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Importancia de las características - XGBoost")
plt.bar(range(X_train_scaled.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train_scaled.shape[1]])
plt.show()


# Una vez finalizados los análisis de todos los modelos, voy a ver los resultados de las dos opciones en una misma celda para analizar mejor la comparativa entre las tablas.

# In[80]:


all_results_todo = results_lasso_todo + results_rf_todo + results_xgb_todo

print_results_table(all_results_todo, "Comparación de los modelos con todas las variables")
print_results_table(all_results, "Comparación de los modelos con selección de características")


# Comprobamos, a priori, que el modelo que obtiene mejores resultados es Random Forest con selección de características o XGBoost con todas las variables, pero vamos a verlo en mayor profundidad en las conclusiones.

# Otra opción también sería comparar cada método de selección de características con cada uno de los modelos Lasso, Random Forest y XGBoost, y ver la combinación más óptima, pero me parece que es computacionalmente muy costoso, por lo que he decidido hacer cada método con su modelo.

# # 4. Conclusiones

# Aquí voy a trasladar las conclusiones de todos los datos obtenidos.

# 1. Lasso no presenta diferencias significativas entre todas las variables y la selección de características, y, en comparación con los otros dos modelos, presenta el MSE y RMSE más alto y menor R2. Por lo que lo descartamos directamente.

# 2. Random Forest mejora ligeramente con la selección de características, donde consigue reducir los errores y mejora también el R2. Su R2 en general es alto, lo que puede reflejar un buen ajuste.

# 3. XGBoost es el único modelo que empeora sus resultados con la selección de características, teniendo mejor ajuste cuando tiene en cuenta todas las variables. Sus diferencias en cuanto a errores y accuracy, respecto a los datos de Random Forest, no son muy significativas.

# Tanto Random Forest como XGBoost parecen buenos modelos que generalizan bien el conjunto de datos. Puede que tengan un ligero sobreajuste debido a la diferencia en los errores de Train y Test, pero, considerando el rango de valores, no parece ser algo especialmente relevante para la generalización.

# Por ello, elijo quedarme con XGBoost, ya que parece generalizar bien, con buenos resultados, parece robusto y podemos considerar ajustes adicionales de sus hiperparámetros como futuros pasos.

# In[ ]:




