# Classification-model-using-K-Nearest-Neighbor-
This notebook builds a classification model using supervised machine learning algorithm

We will be working on a dataset from [dropbox]("https://www.dropbox.com/s/aohbr6yb9ifmc8w/heart_attack.csv?dl=1") with a combination of categorical and numerical variables. The dataset looks at population of different ages and classifies them ob the basis of their liklihood of experiencing heart attack. This dataset has 8 variables including 
1. 'cp' : The type of chest pain they experience 
Value 1: typical angina
Value 2: atypical angina
Value 3: non-anginal pain
Value 4: asymptomatic

2. 'restecg' : resting electrocardiographic results
Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes’ criteria

3. 'thalach' : maximum heart rate achieved during exercise

4. 'output' : the doctor’s diagnosis of whether the patient is at risk for a heart attack

0 = not at risk of heart attack
1 = at risk of heart attack

5. 'age' : Age of the patient 

6. 'sex' : sex of the patient 

7. 'trtbps' : resting blood pressure (in mm Hg)

8. 'chol' : cholesterol in mg/dl fetched via BMI sensor

The data itself was not preprocessed. Variables 'cp' and 'restecg' were catagorical variables with multiple levels. In order to make a robust model, we need to essentially pivot these level to binary forms, or also called 'one-hot encoding' this will allow the model to classify and predict better on the basis of their neighbour. We will use the get_dummies function in pandas library for that. 

## First, Import relevent libraries 
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
%matplotlib inline
```

## Import and explore the data 
```
data = pd.read_csv("heart_attack.csv")
data.info()
data.head()
```
