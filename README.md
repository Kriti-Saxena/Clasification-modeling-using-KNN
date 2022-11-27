# *Classification-model-using-K-Nearest-Neighbor-*

This notebook builds a classification model using supervised machine learning algorithm
We will be working on a dataset from [dropbox](https://www.dropbox.com/s/aohbr6yb9ifmc8w/heart_attack.csv?dl=1) with a combination of categorical and numerical variables. The dataset looks at population of different ages and classifies them ob the basis of their liklihood of experiencing heart attack. This dataset has 8 variables including 

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
from sklearn.metrics import classification_report,confusion_matrix
```

## Import and explore the data 
```
data = pd.read_csv("heart_attack.csv")
data.info()
data.head()
```

Since the 'restecg' and 'cp' is a multiple level categorical variable, we need to pivot them into binary forms also called 'one-hot encoding' using the pandas library 

### cp and rest ecg need to be one-hot encoded into binary form 
```
data_new = pd.get_dummies(data, columns =['restecg'], drop_first = True)
data = pd.get_dummies(data_new, columns=['cp'], drop_first = True)
data
```

## spliting the data into test and train dataset 
```
target = data['output']
inputs = data.drop(['output'], axis =1)
inputs
```
###### since the variables of the data are on different scales, we will normalize the data 
```
# normalising the data 
inputs = (inputs - np.min(inputs))/ (np.max(inputs) - np.min(inputs)).values
```

#### Train test split 
```
X_train, X_test, y_train, y_test = train_test_split(
             inputs, data['output'], test_size = 0.2, random_state=50)
             
```

#### Looking at the accuracy of the model 
```
accuracy = {}
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_test, y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
accuracy['KNN'] = acc
print("Maximum KNN Score is {:.2f}%".format(acc))
```

Here the max accuracy acheved by the model was 83.64%. One can use robust scaling or standard scaling to see how the accuracy differes for different scaling methods

#### using the elbow method and printing confusion matrix 

It is not always easy to figure out the optimal k value, we will use the elbow vizualiser to get the optimal value 

```
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,10))
 
visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
```
[citation](https://towardsdatascience.com/elbow-method-is-not-sufficient-to-find-best-k-in-k-means-clustering-fc820da0631d)

```
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print("roc auc score: ")
print(roc_auc_score(y_test, pred))
print('\n')
print('accuracy score:')
print(accuracy_score(y_test, pred))

```

```
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train,y_train)
predicted_value=clf.predict(x_test)
con_mat=confusion_matrix(y_test,predicted_value)
sns.heatmap(con_mat,annot=True,annot_kws= 
                           {"size":20},cmap="viridis")
plt.show()
```

![Image](./Desktop/confusion matrix.png)
