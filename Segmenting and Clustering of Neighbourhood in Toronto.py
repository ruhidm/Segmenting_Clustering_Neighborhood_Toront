#!/usr/bin/env python
# coding: utf-8

# # Segmenting and Clustering of Neighborgood in Toronto 
# 

# ## Summary of Project
# 
# - 1. **Objective**
# - 2. **Data wrangling**
# - 2.1 Loading data
# - 2.2 Cleaning data
# - 3. **Geolocation and Folium**
# - 3.1 Merging Latitude and Longtidude Data
# - 3.2 Mapping City of Toronto area
# - 3.3 Mapping Downtown Toronto area
# - 3.4 JSON data - shops and business
# - 3.5 JSON data to Pandas DF
# - 3.6 Finding venues and neighborhoods
# - 3.7 Analyzing neighborhoods
# - 4. **Model Evaluation**
# - 4.1 K-Means Clustering
# - 4.2 K-Means Clustering Visualisation
# - 4.3 Examining Clusters
# 

# ## 1. Objective
# 
# Objective of this project is to analyze City of Toronto and Downtown Toronto area shops and business locations. K-Means clustering has been used to determine same business classes in the city area. It is significant to help new starters to choose right location for their business. This projects aid to solve this problem. 

# ## 2. Data Wrangling

# In[1]:


# Importing Essential libraries:

import pandas as pd
import matplotlib as pl
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from pandas import Series, DataFrame
from matplotlib import rcParams
from matplotlib import pyplot

import json

import requests 
from pandas.io.json import json_normalize
from geopy.geocoders import Nominatim 
import folium


# ### 2.1 Loading data

# In[2]:


#Downloading data from Wikipedia
df = pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')[0] # Note: the data can be changed.
df.head()


# In[3]:


# Let's define the shape
df.shape


# ### 2.2 Cleaning data

# As we see from above, we need just one column to show the data. 

# In[4]:


# Let's stack 9 columns to 1 column in order to make it easy to work.
df= pd.DataFrame(df.stack().reset_index(drop = True))
df.head()


# Here we need to drop not assigned values, create three different columns postal, region and neighborhood. We need use lamda and map() function to do that.

# In[5]:


# Data cleaning by using Lambda and map() function.

df.columns = ['all'] # Changing column name to 'all' from 0

df['Postal Code'] = df['all'].map(lambda x:x[0:3]) # separating postal code first tree letters.

df['reg_neig'] = df['all'].map(lambda x:x[3:]) # separating after postal code and making columns

df.drop(df[df['reg_neig'] == 'Not assigned'].index , inplace=True) # dropping Not assigned values

df['region'] = df['reg_neig'].map(lambda x:x.split('(', 1)[0]) # Splitting region from neighborhood

df['Neighborhood'] = df['reg_neig'].map(lambda x:x.split('(', 1)[-1]).str[:-1] # deleting parathesis

df.drop(columns = ['all', 'reg_neig'], inplace = True) # dropping columns
df.head()


# Let's combine them

# In[6]:



print('The dataframe has {} regions and {} neighborhoods.' .format(
        len(df['region'].unique()),
        df.shape[0]))


# Let's create dataframe possessing longtitudes and latitudes

# ## 3. Geolocation and Folium

# ### 3.1 Merging Latitude and Longitude data

# In[7]:


df_lat_long = pd.read_csv('https://cocl.us/Geospatial_data')
df_new = df.merge(df_lat_long, on = 'Postal Code')

df_new.head()


# ### 3.2 Mapping City of Toronto Area

# Each **geolocation** service you might use, such as Google Maps, Bing Maps, or Nominatim, has its own class in geopy.geocoders abstracting the service’s API. Geocoders each define at least a geocode method, for resolving a location from a string, and may define a reverse method, which resolves a pair of coordinates to an address. 
# 
# Let's explore and cluster neighborhood in Toronto

# In[8]:


loc = 'Toronto'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(loc)
latitude = location.latitude
longitude = location.longitude

print('Toronto Latitude is {}, Longitude is {}.' .format(latitude, longitude))


# In[9]:


map_of_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

for lat, lng, region, neighborhood in zip(df_new['Latitude'], df_new['Longitude'], df_new['region'], df_new['Neighborhood']):
    label = '{}, {}'.format(neighborhood, region)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_of_toronto)  
    
map_of_toronto


# In[10]:


# Counting region values
df_new['region'].value_counts()


# ### 3.3  Mapping Downtown Toronto area

# In[11]:


downtown_toronto = df_new[df_new['region'] == 'Downtown Toronto'].reset_index(drop=True)
downtown_toronto.head()


# In[12]:


loc1 = 'Downtown Toronto'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(loc1)
latitude = location.latitude
longitude = location.longitude
print('Toronto Latitude is {}, Longitude is {}.' .format(latitude, longitude))


# In[13]:


map_downtown_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

for lat, lng, region, neighborhood in zip(downtown_toronto['Latitude'], downtown_toronto['Longitude'], downtown_toronto['region'], downtown_toronto['Neighborhood']):
    label = '{}, {}'.format(neighborhood, region)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_downtown_toronto)  
    
map_downtown_toronto


# In[14]:


# Let's explore the neighborhood and segment them
downtown_toronto.loc[0, 'Neighborhood']


# In[15]:


neighbor_latitude = downtown_toronto.loc[0, 'Latitude']
neighbor_longitude = downtown_toronto.loc[0, 'Longitude']

neighbor_name = downtown_toronto.loc[0, 'Neighborhood']

print('Latitude and longitude values of {} are {} and {}' .format(neighbor_name, 
                                                               neighbor_latitude, 
                                                               neighbor_longitude)
     )


# ### 3.4 JSON Data - Shops

# In[16]:


CLIENT_ID = 'TOITK2RBLZUXQDMPORAIE5GARRN2E1RRHN0RS2EGNJ1QSL3L'
CLIENT_SECRET = 'RMUMYUEAQTEYRGZBFWSQVVAWD5Z4DM2H4XJPH2FHFQONI3EV'
VERSION = '20180605'
LIMIT = 100
radius = 500

url_venues = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighbor_latitude, 
    neighbor_longitude, 
    radius, 
    LIMIT)

url_venues


# **JSON** (JavaScript Object Notation) is an open standard file format, and data interchange format, that uses human-readable text to store and transmit data objects consisting of attribute–value pairs and array data types (or any other serializable value).

# In[17]:


results = requests.get(url_venues).json() # loading JSON data
results


# In[18]:


# Let's define the function
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# ### 3.5 JSON data to Pandas DF

# In[19]:


# Let's structure json data to Pandas Dataframe
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues)

filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# ### 3.6 Finding venues and neighberhoods

# In[20]:


print('{} venues were returned by Foursquare near {}' .format(nearby_venues.shape[0], neighbor_name))


# In[21]:


# Let's repeat the all process in DT Toronto
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    LIMIT = 100
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url_foursquare = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url_foursquare).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])
        
    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[22]:


df_venues = getNearbyVenues(names=downtown_toronto['Neighborhood'],
                            latitudes=downtown_toronto['Latitude'],
                            longitudes=downtown_toronto['Longitude']
                            )


# In[23]:


df_venues.shape


# In[24]:


df_venues.head()


# #### Let's count the neighberhoods which were returned

# In[25]:


df_venues.groupby('Neighborhood')['Venue'].count().to_frame()


# In[26]:


print('There are {} unique categories.' .format(len(df_venues['Venue Category'].unique())))


# ### 3.7 Analysing Neighborhood 

# In[27]:


# Analysing neighborhood
# one hot encoding
toronto_onehot = pd.get_dummies(df_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = df_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[28]:


toronto_onehot.shape


# In[29]:


# groupby() by Neighborhood
df_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
df_grouped.head()


# In[30]:


num_top_venues = 5

for hood in df_grouped['Neighborhood']:

    temp = df_grouped[df_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# Create a new dataframe and display the top 10 venues for each neighborhood

# In[31]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[32]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = df_grouped['Neighborhood']

for ind in np.arange(df_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(df_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## 4. Model Evaluation

# ### 4.1 K-Means Clustering

# In[33]:


from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

toronto_grouped_clustering = df_grouped.drop('Neighborhood', 1)

# run k-means clustering
KM = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
KM.labels_[0:10]


# In[34]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', KM.labels_)

toronto_merged = downtown_toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# ### 4.2 K-Means Clustering Visualisation

# In[35]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ### 4.3 Examining clusters

# #### Cluster 1

# In[36]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# #### Cluster 2

# In[37]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# #### Cluster 3

# In[38]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# #### Cluster 4

# In[39]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# #### Cluster 5

# In[40]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:




