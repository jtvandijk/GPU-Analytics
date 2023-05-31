---
title: "Chapter 3 - Geospatial Operation and Analysis" 
weight: 1
# bookFlatSection: false
# bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: false
# bookSearchExclude: false
---


## Geospatial Operation and Analysis 

Why are we interested in geospatial data? Geospatial data is a type of data that is associated with a location. This location can be a point, a line, a polygon, or a raster. Geospatial data is becoming more and more important, yet the sheer volume of information that is generated made it difficult to handle. In this chapter, we will be exploring the use of GPU for manipulating large-scale geospatial data, and provide a practical example of how we can predict the presence of Aedes aegypti across the globe. 

## Objectives 

The objective of the third *Case Study* is to demonstrate the practical application of the common spatial data structures and operation with GPU-accelerated Python packages. The goal here is in twofold: 1) compare the computational speed of calculating point in polygon and 2) compare the computational speed of a classification model with raster data. 

## Predicting the Presence of Aedes aegypti 

In this case study, we will be predicting the global presence and probability of Aedes aegypti, a mosquito species that is known to transmit dengue fever, chikungunya, and Zika virus. You can download the Aedes aegypti point occurrence data across the World from 1958 to 2014 [here](https://github.com/jasoncpit/GPU-Analytics/blob/master/data/Chapter3/aedes_point.csv). We will also be using precipitation, temperature, elevation, and population density as predictor variables to capture the climatic, environmental and demographics variables. You can download the raster data from the GitHub repository [here](https://github.com/jasoncpit/GPU-Analytics/tree/master/data/Chapter3).

### Load datasets 
To begin, let's load the necessary libraries and datasets. It is important to note that all shape file and raster data (5km resolution) were projected to the CRS: WGS84 4236.
```python
# Import libraries
import rasterio
import geopandas as gpd 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import cuspatial
import cudf

# Load raster data 
precipitation = rasterio.open('./data/Global Raster/Precipitation/Global Averaged Precipitation 2000-14.tif')
temp = rasterio.open('./data/Global Raster/Temperature/Global Averaged Temperature 2000-14.tif')
elevation = rasterio.open('./data/Global Raster/Elevation/Global Land Surface Elevation.tif')
pop_density = rasterio.open('./data/Global Raster/Population Density/Global Population Density AveragedEst.tif')

# Load Shapefile --------------------------------
# Loading global shapefile 
global_outline =gpd.read_file('./data/Global Shapefile/Global Outline/Global_All_0.shp',crs='EPSG:4326')
# Loading country shapefiles 
country_outline = gpd.read_file('./data/Global Shapefile/Global with Country Outline/Global_Countries_1.shp',crs='EPSG:4326') 

# Load Point occurrence data 
aedes =pd.read_csv('./data/Global Raster/aedes_point.csv')
aedes_point = gpd.GeoDataFrame(aedes,geometry=gpd.points_from_xy(aedes['longitude'],aedes['latitude']),crs='EPSG:4326')

# Transformation 
global_outline.crs = ('+init=EPSG:4326')
country_outline.crs =('+init=EPSG:4326')  
aedes_point.crs = ('+init=EPSG:4326')

# Check CRS 
print(global_outline.crs ==aedes_point.crs)
```

### Data Visualization 

Let's visualize the point occurrence data and the predictor variables. We will be using the `matplotlib` and `rasterio` packages to visualize the data. 
```python
# Visualise point patterns 
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1,1, figsize = (10,10)) 
country_outline.plot(edgecolor='black',linewidth=0.1,ax=ax,color="white")
aedes_point.plot(ax=ax,color='red',markersize = 1) 
ax.axis('off')
ax.set_title("Distribution of aedes occurence across the World")
```

<figure title = "Global Distribution">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_global.png?raw=true">
    <figcaption>
    <b>Figure 1: Global Distribution of Aedes aegypti 
    </b>
    </figcaption>
    </center>
</figure>

```python
from rasterio.plot import show 
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(20,20))
title =["precipitation","temp","elevation","pop_density"] 
for index,attribute in enumerate([precipitation,temp,elevation,pop_density]): 
     image = show(attribute,ax=ax[index//2,index%2],cmap='nipy_spectral',title = title[index])
     image.axis('off')
fig.subplots_adjust(hspace=-0.5, wspace=0)
```

<figure title = "Predictors Distribution">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_predictors.png?raw=true">
    <figcaption>
    <b>Figure 2: Distribution of Predictors: Precipitation, Temperature, Elevation, and Population Density 
    </b>
    </figcaption>
    </center>
</figure>


### Prepare data for pseudo-background points as absence 

Next, we need to prepare the background data. What is the background data? With background data we are not attempting to guess point locations where an event is absent. Here, we are rather trying to characterise the environment of the study region. In this sense, background is the same, irrespective of where the point fire are found or not. Background data establishes the environmental domain of the study, whilst presence data should establish under which conditions a fire is more likely to be present than on average. 

There are several ways to generate background data. In R, we can use the `spsample()` function from the `sp` package to generate randomly-distributed points with defined spatial boundaries. However, in Python, there is no pre-built function to help us. Instead, we will generate random points across the world and filter out the points that are outside the country boundary. Here is the code to generate random points across the world.

```python
# random seed 
import random
from shapely.geometry import Point
from tqdm import tqdm
random.seed(10)
def Random_Points_in_Polygon(polygon, number):
    bound = polygon.bounds.values[0]
    minx, miny, maxx, maxy = bound[0],bound[1],bound[2],bound[3]
    x_point,y_point = np.random.uniform(minx, maxx,size=number),np.random.uniform(miny,maxy,size=number) 
    return x_point,y_point
background_points = Random_Points_in_Polygon(global_outline,aedes_point.shape[0]*5)
background_points_shp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=background_points[0],y=background_points[1]),crs='EPSG:4326')
background_points_shp.crs = global_outline.crs 
print("Number of background points: ",background_points_shp.shape[0]) 
#Number of background points:  171108 
```

Figure 3 shows the distribution of background points across the world. We generated 171,108 points that are evenly distributed across the world, with some points located in the ocean. 

```python
# Visualise point patterns 
fig, ax = plt.subplots(1,1, figsize = (10,10)) 
global_outline.plot(edgecolor='black',linewidth=2,ax=ax,color="white")
background_points_shp.plot(ax=ax,color='blue',markersize = 0.1) 
ax.axis('off')
ax.set_title("Distribution of background points across the World")
```


<figure title = "Predictors Distribution">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_background_point.png?raw=true">
    <figcaption>
    <b>Figure 3: Generating pseudo-background points as absence 
    </b>
    </figcaption>
    </center>
</figure>


### Calculate point in polygon with GeoPandas and CuSpatial

Now that we've created the background points, we need to quickly determine which points occur in each country. This task is commonly known as a Point in Polygon (PIP) query, however, they are usually extremely slow. In Python, we can use the Shapely library's functions `.within()` and `.contains()` to determine if a point is within a polygon or if a polygon contains a point. Nevertheless, these functions are not designed to handle large datasets. Assuming that we have 1 million points and 100 polygons, the `.within()` function will take 100 million comparisons to determine which points are within which polygons. This is a very slow process.

Let's first use the `.sjoin()` function from GeoPandas to determine which points are within which polygons. Using the `%%time` magic function, we can see that the `.sjoin()` function takes 8.6 seconds to compute 171108 points. 

```python
# pip query with geopandas 
%%time 
pointInPolys = gpd.sjoin(background_points_shp, country_outline) 
#CPU times: user 8.6 s, sys: 43 ms, total: 8.65 s
#Wall time: 8.69 s
```

Next, we can use the `.point_in_polygon` function from the GPU-accelerated library `cuSpatial`. It is important to note here that the `.point_in_polygon` requires the points to be in the form of a `cudf.DataFrame` and the polygons to be in the form of a `cuspatial.GeoSeries`. Here, we are using the `cuspatial.read_polygon_shapefile()` function to make sure the resulting tuple `poly_offsets`, `poly_ring_offsets`, `poly_points` perfectly matches the input requirements of `point_in_polygon`. 

Because the `.point_in_polygon` function can only handle 31 polygons at a time, we need to split the polygons into batches of 31 polygons. The for loop shown below iterates through each batch and append true values in the array to a new Country ID, matching the spatial indices of the polygons.

The `.point_in_polygon` function performs the PIP query in 0.5 seconds, which is 16 times faster than the `.sjoin()` function. 

```python
%%time
# Cuspatial 
cu_countries_outline = cuspatial.read_polygon_shapefile('./data/Global Shapefile/Global with Country Outline/Global_Countries_1.shp')

background_points_df = cudf.DataFrame() 
background_points_df['Long'] = background_points[0]
background_points_df['Lat'] = background_points[1]
background_points_df['LocationID'] =  cu_countries_outline[0].shape[0] 
pip_iterations = list(np.arange(0, cu_countries_outline[0].shape[0], 31))

for i in range(len(pip_iterations)-1):
    start = pip_iterations[i]
    end = pip_iterations[i+1]
    pip_countries = cuspatial.point_in_polygon(background_points_df['Long'],background_points_df['Lat'],cu_countries_outline[0][start:end],cu_countries_outline[1],cu_countries_outline[2]['x'],cu_countries_outline[2]['y'])
    for j in pip_countries.columns:
        background_points_df['LocationID'].loc[pip_countries[j]] = j
#CPU times: user 553 ms, sys: 8.88 ms, total: 562 ms
#Wall time: 563 ms         
```

### Stress Test 

The differences may be marginal at this point, but the `.point_in_polygon` function will become more significant as the number of background points increases. We can run a stress test by increasing the number of background points. The code below generates 10-30 times more background points than the original dataset. We can see from Figure 4 that the `.sjoin()` function takes linearly longer to compute the PIP query, while the `.point_in_polygon` function takes a constant time to compute the PIP query, with average runtimes of less than 2.5 seconds.  

```python
import time
cpu_time = []
gpu_time = []

for i in range(10,30):
    #Generate random background points
    background_points = Random_Points_in_Polygon(global_outline, aedes_point.shape[0] * i)

    #Preparing data on Geopandas 
    background_points_shp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=background_points[0], y=background_points[1]), crs='EPSG:4326')
    background_points_shp.crs = global_outline.crs
    #cpu runtime -------------------------------- 
    start = time.time()
    CPU_pointInPolys = gpd.sjoin(background_points_shp, country_outline)
    end = time.time()
    cpu_time.append(end - start)

    #Preparing data on cudf 
    background_points_df = cudf.DataFrame()
    background_points_df['Long'] = background_points[0]
    background_points_df['Lat'] = background_points[1]
    background_points_df['LocationID'] = cu_countries_outline[0].shape[0]
    pip_iterations = list(np.arange(0, cu_countries_outline[0].shape[0], 31))

    #gpu runtime --------------------------------
    start = time.time()
    for iter in range(len(pip_iterations) - 1):
        pip_start = pip_iterations[iter]
        pip_end = pip_iterations[iter + 1]
        pip_countries = cuspatial.point_in_polygon(background_points_df['Long'], background_points_df['Lat'], cu_countries_outline[0][pip_start:pip_end], cu_countries_outline[1], cu_countries_outline[2]['x'], cu_countries_outline[2]['y'])
        for j in pip_countries.columns:
            background_points_df['LocationID'].loc[pip_countries[j]] = j
    end = time.time()
    gpu_time.append(end - start)

```

```python 
#Create a dataframe to store the results
gpu_elapsed = pd.DataFrame({"time":gpu_time,"data_size":[aedes_point.shape[0]* i for i in range(10,30)],'label':"cuspatial.point_in_polygon"})
cpu_elapsed = pd.DataFrame({"time":cpu_time,"data_size":[aedes_point.shape[0]* i for i in range(10,30)],'label':"gpd.sjoin"})
result = pd.concat([gpu_elapsed,cpu_elapsed]).reset_index()

#Plot results
fig, ax = plt.subplots(figsize=(10,10))
sns.lineplot(x= 'data_size',y='time',hue = 'label',data = result,ax = ax )
plt.xlabel('Data Size')
plt.ylabel("Time Elapsed ")
plt.title("Comparing the speed of Point-in-Polygons calculation on CPU and GPU") 
plt.show()
```


<figure title = "PIP SPEED">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_pip.png?raw=true">
    <figcaption>
    <b>Figure 4: Comparing the speed of Point-in-Polygons calculation on CPU and GPU 
    </b>
    </figcaption>
    </center>
</figure>


### Visualize point patterns 

Now that we know how to calculate the PIP query, we can visualize the point patterns by selecting the background points that are within the polygon of a country. The code below visualizes the point patterns of the countries with the highest number of cases. 

```python
# Visualise point patterns

#Recalculate the PIP query with Cuspatial 
background_points = Random_Points_in_Polygon(global_outline,aedes_point.shape[0]*5)
background_points_df = cudf.DataFrame() 
background_points_df['Long'] = background_points[0]
background_points_df['Lat'] = background_points[1]
background_points_df['LocationID'] =  cu_countries_outline[0].shape[0] 
pip_iterations = list(np.arange(0, cu_countries_outline[0].shape[0], 31))

for i in range(len(pip_iterations)-1):
    start = pip_iterations[i]
    end = pip_iterations[i+1]
    pip_countries = cuspatial.point_in_polygon(background_points_df['Long'],background_points_df['Lat'],cu_countries_outline[0][start:end],cu_countries_outline[1],cu_countries_outline[2]['x'],cu_countries_outline[2]['y'])
    for j in pip_countries.columns:
        background_points_df['LocationID'].loc[pip_countries[j]] = j

pointInPolys = background_points_df.query("LocationID != 250")  
pointInPolys = gpd.GeoDataFrame(geometry = gpd.points_from_xy(pointInPolys['Long'].to_numpy(),pointInPolys['Lat'].to_numpy()),crs="EPSG:4326")
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1,1, figsize = (10,10)) 
global_outline.plot(edgecolor='black',linewidth=1,ax=ax,color="white")
pointInPolys.plot(ax=ax,color='blue',markersize = 0.1) 
ax.axis('off')
ax.set_title("Distribution of background points across Brazil")
```


<figure title = "Predictors Distribution">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_filtered_background.png?raw=true">
    <figcaption>
    <b>Figure 5: Filtering the pseudo-background points that are within the country boundary
    </b>
    </figcaption>
    </center>
</figure>


### Create a multi-band raster from the predictor variables 

To facilitate the required analysis, it is necessary to construct a multi-band raster object from the predictor variables. A band is essentially a matrix of cell values, and a raster with multiple bands comprises multiple matrices of cell values that overlap spatially and represent the same geographic region. For example, the raster object for temperature is a single-band raster object. However, if we stack raster objects for variables such as precipitation, population density and elevation on top of the temperature raster, we create a multi-band raster object. This object will have four bands, each corresponding to a single matrix of cell values.

The creation of this multi-band raster object is essential to perform the extraction of raster values from all variables at occurrence points in a single step. Additionally, the entire multi-band raster object is required for estimating and predicting spatial data.

```python
meta = precipitation.meta
meta.update(count = 4)
with rasterio.open('./data/Global Raster/global_stack.tif', 'w',**meta) as dst:
    for index,attribute in enumerate([precipitation,temp,elevation,pop_density],start=1): 
        dst.write(attribute.read(1),index) 
stack = rasterio.open('./data/Global Raster/global_stack.tif')         
```


### Extraction of all raster values from predictor variables onto presence-absence points 

Now, we are going to extract information from our raster stack to both the presence and background points. This can be done using the sample function in rasterio. For all occurrence points (i.e., presence), we need to add an indicator of 1 to signify presence; while for all background points (i.e., absence) - we need to also add an indicator of 0 to signify absence. We do this because we are modelling a probability and such niche models take outcomes that are from a Bernoulli or Binomial distribution.

```python
# Extrat raster values 
background_list = [(x,y) for x,y in zip(pointInPolys['geometry'].x , pointInPolys['geometry'].y)]
pointInPolys['value'] = [x for x in stack.sample(background_list)]

aedes_list = [(x,y) for x,y in zip(aedes_point['geometry'].x , aedes_point['geometry'].y)]
aedes_point['value'] = [x for x in stack.sample(aedes_list)]

# Convert into dataframe 
aedes_env = pd.DataFrame(np.vstack(aedes_point['value']),columns=title)
aedes_env['Presence'] = 1
background_env = pd.DataFrame(np.vstack(pointInPolys['value']),columns=title)
background_env['Presence'] = 0 

#Merge 
input_data = pd.concat([aedes_env,background_env],axis=0)
input_data[input_data<0]  =0
```

### Preparation of training & test data for prediction & model cross-validation

Now, we need to prepare our data for model cross-validation. We are going to split our data into training and test data. The training data will be used to train the model, while the test data will be used to validate the model. The test data will be used to assess the model’s performance on data that it has not seen before. We are going to use a 80:20 split for training and test data, respectively.

```python
#Split train,test set 
from sklearn.model_selection import train_test_split
y= input_data.pop('Presence')
X = input_data 
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8)
```

### Random Forest Classification Model with CuML and Scikit-learn

Now, we can fit the random forest model, which tries to find the combination of environmental risk factors that best predicts the occurrence of the aedes aegypti. 

A random forest is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. For more information on random forest, check [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

For demonstration purposes, we are going to use the random forest model from both Scikit-learn and cuML. cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives functions on GPU. cuML is designed to be a drop-in replacement for Scikit-learn. For more information on cuML, check [here](https://docs.rapids.ai/api/cuml/stable/).

As we can see the code below, the cuML random forest model is very similar to the Scikit-learn random forest model. The only difference is that we need to convert the training and test data into CuPy arrays. 

Overall, we can see that the cuML random forest model is much faster than the Scikit-learn random forest model, with a speedup of from 4.65s to 0.34s.

```python
%%time 
# Random forest with sklearn 
from sklearn.ensemble import RandomForestClassifier
sk_model = RandomForestClassifier(max_depth=20, random_state=42,n_estimators=100)
sk_model.fit(X_train,y_train)
y_pred =sk_model.predict(X_test) 
#CPU times: user 4.63 s, sys: 9.95 ms, total: 4.64 s
#Wall time: 4.65 s
```  

```python
%%time 
# Random forest with cuml 
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRFC
cuml_model = cuRFC(max_depth=20, random_state=42,n_estimators=100)
cuml_model.fit(cp.array(X_train),cp.array(y_train)) 
cuml_predict = cuml_model.predict(cp.array(X_test))
#CPU times: user 2.81 s, sys: 359 ms, total: 3.17 s
#Wall time: 341 ms
``` 

### Model validation and examination of the predictor’s contribution 

Now that we have fitted the random forest model, we can examine the model’s performance. We can do this by the model’s accuracy, the area under the curve (AUC) and `max TPR+TNR`. The AUC is a measure of the model’s performance. The higher the AUC, the better the model is at distinguishing between the presence and absence of the aedes aegypti. The `max TPR+TNR` denotes the probability threshold at which our model maximizes the True Positive Rate and the True Negative Rate. It is generally accepted that this is an optimum value at which to set the threshold for binary classification of the predicted probabilities in our mapping outputs. Anything above value is deemed as a region environmentally suitable for outcome. 

```python 
# Model validation
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,cp.asnumpy(cuml_predict)))
#Accuracy: 0.9372446306966998
from sklearn.metrics import RocCurveDisplay

y_pred_proba = cuml_model.predict_proba(X_test)[::,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr)
plt.legend(loc=4)


plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC = {}".format(auc))
plt.legend()
plt.show()

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Max TRP+TNR:", optimal_threshold)
#Max TRP+TNR: 0.36108708

```

From Figure 6, we can see that the AUC of the random forest model is amost 0.99, which is substantially higher than the baseline of 0.5. This means that the random forest model is able to distinguish between the presence and absence of Aedes aegypti with a high degree of accuracy.  The optimal probability threshold at which our model maximizes the True Positive Rate and the True Negative Rate is 0.3611 (36.1%). Hence, we will use predicted probability > 0.3611 to delineate areas of suitability (or trigger points) for the presence of Aedes aegypti. 


<figure title = "Predictors Distribution">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_ROC_curve.png?raw=true">
    <figcaption>
    <b>Figure 6: ROC Curve of the Random Forest Classification Model
    </b>
    </figcaption>
    </center>
</figure>


We can also examine the feature importance of the random forest model. The feature importance is a measure of how much each predictor contributes to the model’s performance. The higher the feature importance, the more important the predictor is in predicting the outcome. 

```python
# Feature importance 
import pandas as pd
feature_imp = pd.Series(sk_model.feature_importances_,index=title).sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
``` 

From Figure 7, we can see that population density is the most important predictor in predicting the presence of Aedes aegypti. However, it is important to note that cuML does not support the feature importance function. 

<figure title = "Predictors Distribution">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_Feature_importance.png?raw=true">
    <figcaption>
    <b>Figure 7: Feature importance of the Random Forest Classification Model
    </b>
    </figcaption>
    </center>
</figure>


### Mapping the predicted probability of presence of Aedes aegypti 

To map the predicted probability of presence of Aedes aegypti, we need to predict the probability of presence of Aedes aegypti for each grid cell. To do this, we need to convert the environmental matrices into a 2D array. We can then use the `predict_proba` function from `sklearn` and `cuML` to predict the probability of presence of Aedes aegypti for each grid cell. 

Again, we can see a significant improvement in the speed of the prediction using `cuML` compared to `sklearn`, reducing the prediction time from 2min 51s to 2.06s.  

```python
# Mapping predicted probability and suitability
input_matrix = stack.read()  
#input shape (4, 4320, 8640)
#Convert into (4, 4320*8640)
converted_env = [] 
for i in range(4): 
    attr = input_matrix[i,:,:].reshape(input_matrix.shape[1]*input_matrix.shape[2],1)
    converted_env.append(attr)

converted_env = pd.DataFrame(np.hstack(converted_env),columns=title)
converted_env[converted_env<0] =0
```  


```python 
%%time
# Predict the probability of presence of Aedes aegypti for each grid cell with sklearn
y_pred =sk_model.predict_proba(converted_env)[::,-1] 
#CPU times: user 2min 29s, sys: 22.1 s, total: 2min 51s
#Wall time: 2min 51s

```

```python
%%time 
# Predict the probability of presence of Aedes aegypti for each grid cell with cuML
y_pred =cuml_model.predict_proba(converted_env)[::,-1]
#CPU times: user 1.73 s, sys: 871 ms, total: 2.6 s
#Wall time: 2.06 s
```

We can then convert the predicted probability of presence of Aedes aegypti into a raster file. Here, we converted probability estimate with less than 0.3611 as 0 and anything above as 1. The predicted probability > 0.3611 4 are the areas that are expected to have the presence of Aedes aegypti.

```python
# Convert the predicted probability of presence of Aedes aegypti into a raster file
with rasterio.open('./data/Global Raster/prediction.tif', 'w',**meta) as dst:
     # convert to numpy array if the prediction is from cuML
    dst.write(cp.asnumpy(y_pred).reshape((input_matrix.shape[1],input_matrix.shape[2])),1)


# Generate final output which shows grid cells with probability > 0.3611 
trigger = y_pred
trigger[trigger >=0.36108708]  = 1
trigger[trigger < 0.36108708]  = 0


with rasterio.open('./data/Global Raster/trigger.tif', 'w',**meta) as dst:
     # convert to numpy array if the prediction is from cuML
    dst.write(cp.asnumpy(trigger).reshape((input_matrix.shape[1],input_matrix.shape[2])),1) 

prediction = rasterio.open('./data/Global Raster/prediction.tif') 
trigger = rasterio.open('./data/Global Raster/trigger.tif')        
```

We can then map the predicted probability of presence of Aedes aegypti and trigger points. 

```python
fig,ax= plt.subplots(2,1,figsize=(20,20))
country_outline.boundary.plot(edgecolor='white',linewidth=0.5,ax=ax[0])
show(prediction,cmap='nipy_spectral',ax=ax[0])

country_outline.boundary.plot(edgecolor='white',linewidth=0.5,ax=ax[1])
show(trigger,cmap='nipy_spectral',ax=ax[1])

ax[0].set_axis_off()
ax[1].set_axis_off()
ax[0].set_title('Predicted Probability of Presence of Aedes aegypti',fontsize=20)
ax[1].set_title('Trigger Points',fontsize=20)
plt.show()
```

<figure title = "Predictors Distribution">
     <center>
     <p><img src="https://github.com/jasoncpit/GPU-Analytics/blob/master/Pictures/chp3_maps.png?raw=true">
    <figcaption>
    <b>Figure 8: Predicted probability of presence of Aedes aegypti and trigger points 
    </b>
    </figcaption>
    </center>
</figure>

### Conclusion 

In this chapter, we demonstrated an end-to-end workflow of using GPU to 1) calculate the point in polygon function, 2) train a random forest classification model, and 3) predict the probability of presence of Aedes aegypti. In all three steps, using a GPU for working with raster files in Python can offer significant performance improvements. 

What we have shown here is just a small fraction of what we can do with GPU. More importantly, the functions provided here can be easily migrated into existing workflows where the data is too large to be processed on a CPU. 


