

```python
##House Prices: Advanced Regression Techniques Test 
##Author: Nok Chan 
##Last modified: 8/13/2017
```

![Kaggle House Price Competition](HousePrice.png)

![Rank](rank1.png)

![Total Participant](rank2.png)


```python
%config IPCompleter.greedy=True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble, tree, linear_model
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import os
%matplotlib inline 
#This line force the graph print out in this jupyter Notebook
```


```python
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
```


```python

testset = pd.read_csv('test.csv')
trainset = pd.read_csv('train.csv')
```


```python
trainset.isnull().sum()
## Some features have almost all null in every rows, so I will remove null> 1000 for cleaning the features a little bit.

```




    Id                  0
    MSSubClass          0
    MSZoning            0
    LotFrontage       259
    LotArea             0
    Street              0
    Alley            1369
    LotShape            0
    LandContour         0
    Utilities           0
    LotConfig           0
    LandSlope           0
    Neighborhood        0
    Condition1          0
    Condition2          0
    BldgType            0
    HouseStyle          0
    OverallQual         0
    OverallCond         0
    YearBuilt           0
    YearRemodAdd        0
    RoofStyle           0
    RoofMatl            0
    Exterior1st         0
    Exterior2nd         0
    MasVnrType          8
    MasVnrArea          8
    ExterQual           0
    ExterCond           0
    Foundation          0
                     ... 
    BedroomAbvGr        0
    KitchenAbvGr        0
    KitchenQual         0
    TotRmsAbvGrd        0
    Functional          0
    Fireplaces          0
    FireplaceQu       690
    GarageType         81
    GarageYrBlt        81
    GarageFinish       81
    GarageCars          0
    GarageArea          0
    GarageQual         81
    GarageCond         81
    PavedDrive          0
    WoodDeckSF          0
    OpenPorchSF         0
    EnclosedPorch       0
    3SsnPorch           0
    ScreenPorch         0
    PoolArea            0
    PoolQC           1453
    Fence            1179
    MiscFeature      1406
    MiscVal             0
    MoSold              0
    YrSold              0
    SaleType            0
    SaleCondition       0
    SalePrice           0
    Length: 81, dtype: int64




```python
features = trainset.columns.values
remove_features = trainset.columns[trainset.isnull().sum()>1000]

```


```python
print(testset.shape)
print(trainset.shape)
```

    (1459, 80)
    (1460, 81)
    


```python
trainset = trainset.drop(remove_features,1)
trainset = trainset.drop('Id', 1)

```


```python
testset = testset.drop(remove_features,1)
testset_id = testset['Id']
testset = testset.drop('Id', 1)
```


```python
remove_features
```




    Index(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], dtype='object')




```python
print(testset.shape)
print(trainset.shape)
```

    (1459, 75)
    (1460, 76)
    


```python
## For now, i will start with the numerical variable and ignore the categorial variables for a while
## Start with correlation always give you some insights about how different factor related.
## Since the number of variables is large, it's not easy to visualize with table, a matrix could help us on this.
## Seaborn library-
train_cor = trainset.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(train_cor,vmax=0.1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1de964d1080>




![png](output_12_1.png)



```python
## From the heatmap we can see some variable has correlation very close to 0.8
## Grage YrBlt,GarageCars,Garage Area are strongly correlated, we don't want this collinearity as Area of Garage 
##is limiting factor of how many car u can park.

## TotalBsmtSF and ,1stFlrSF  is also highly correlated, see the metadata.
##     1stFlrSF: First Floor square feet
## TotalBsmtSF: Total square feet of basement area

## YearBLT is also highly correlated to GarageYearBLT
## OverallQuality has a strong correlation with Sales Price too, this is good as this is a useful indicator for predicting 
## sales price which is our ultimate goal.


```


```python
## To decide which factor we should keep, I will keep the factor that are more correlated to sales price as I think this
## will help the performence of the model.

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = train_cor.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(trainset[cols].values.T)
sns.set(font_scale=1.25)


hm = sns.heatmap(cm,cmap="RdBu_r", yticklabels=cols.values, xticklabels=cols.values)

## The order of the features is already sorted in descending order.



```


![png](output_14_0.png)



```python
## GarageCars has a higher correlation, so Garage Area and GarageYearBlt will be dropped.
## TotalBsmtSF has a higher correlation with 1stFlrSF, so it will be dropped as well.
trainset = trainset.drop(['1stFlrSF','GarageArea','GarageYrBlt'],1)
testset = testset.drop(['1stFlrSF','GarageArea','GarageYrBlt'],1)
```


```python
print(trainset.shape)
print(testset.shape)
## Ok , 3 mores columns dropped
```

    (1460, 73)
    (1459, 72)
    


```python
## So some more EPA about our target variable -- Sales Price

sns.distplot(trainset['SalePrice'])


## The data looks quite good, money is always appear in a skewed distribution in normal scale, we can normalized it in a log scale.

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1de966439e8>




![png](output_17_1.png)



```python
sns.distplot(np.log(trainset['SalePrice']))

## Voila, it is very close to normal distrubtion.
## Therefore, it is good to apply log transformation to SalePrice variable as regression usually need a normal distribution error assumption

trainset['SalePrice'] =np.log(trainset['SalePrice'])
```


![png](output_18_0.png)



```python
train_labels=trainset['SalePrice']
trainset=trainset.drop('SalePrice',1)
```


```python
print(trainset.shape)
print(testset.shape)
## Ok , 3 mores columns dropped
```

    (1460, 72)
    (1459, 72)
    


```python
trainset_index = range(len(trainset))
testset_index = range(len(trainset),len(trainset) + len(testset))
```


```python
masterset = pd.concat([trainset,testset], axis = 0)
```


```python
masterset.shape
```




    (2919, 72)




```python
masterset_backup = masterset
# Getting Dummies from all other categorical vars
for col in masterset.dtypes[masterset.dtypes == 'object'].index:
    for_dummy = masterset.pop(col)
    masterset = pd.concat([masterset, pd.get_dummies(for_dummy, prefix=col)], axis=1)
```


```python
masterset.shape
```




    (2919, 272)




```python
train_features = masterset.iloc[trainset_index] 
test_features = masterset.iloc[testset_index]
```


```python
print(train_features.shape)
print(test_features.shape)
```

    (1460, 272)
    (1459, 272)
    


```python
##seperator
```


```python
train_features= trainset
# Getting Dummies from all other categorical vars
for col in train_features.dtypes[train_features.dtypes == 'object'].index:
    for_dummy = train_features.pop(col)
    train_features = pd.concat([train_features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
```


```python
test_features = testset
for col in test_features.dtypes[test_features.dtypes == 'object'].index:
    for_dummy = test_features.pop(col)
    test_features = pd.concat([test_features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
```


```python
print(test_features.shape)
print(train_features.shape)
```

    (1459, 242)
    (1460, 255)
    


```python
train_features.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 272 columns</p>
</div>




```python
test_features.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>80.0</td>
      <td>11622</td>
      <td>5</td>
      <td>6</td>
      <td>1961</td>
      <td>1961</td>
      <td>0.0</td>
      <td>468.0</td>
      <td>144.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>81.0</td>
      <td>14267</td>
      <td>6</td>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>108.0</td>
      <td>923.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>74.0</td>
      <td>13830</td>
      <td>5</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>0.0</td>
      <td>791.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>78.0</td>
      <td>9978</td>
      <td>6</td>
      <td>6</td>
      <td>1998</td>
      <td>1998</td>
      <td>20.0</td>
      <td>602.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120</td>
      <td>43.0</td>
      <td>5005</td>
      <td>8</td>
      <td>5</td>
      <td>1992</td>
      <td>1992</td>
      <td>0.0</td>
      <td>263.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 256 columns</p>
</div>




```python
## Model
GBest = ensemble.GradientBoostingRegressor( max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')
Pipe = Pipeline([
        ('imp', Imputer(missing_values='NaN')),
        ('std', MinMaxScaler()),
        ('selection', SelectKBest()),
        ('pca', PCA()),
        ('boost', GBest)
    ])
# estimator parameters
kfeatures = [128,256]
components = [4,8,16,32,64]
estimators = [64,128,256,512,1024,2048,4096]
learnrate = [0.001,0.01,0.1,1]
depth = [3,6,9]

param_grid={'selection__k': kfeatures,
              'pca__n_components': components,
              'imp__strategy': ['mean','median','most_frequent'],
                'imp__missing_values': ['NaN'],
              'boost__n_estimators': estimators,
              'boost__learning_rate':  learnrate,
            'boost__max_depth' : depth
               }
```


```python
# set model parameters to grid search object
gridCV_object = RandomizedSearchCV(estimator = Pipe, 
                             param_distributions = param_grid,
                             n_iter = 100)
                             #scoring = scorer,
                             #cv = StratifiedShuffleSplit(n_splits = 10, test_size=0.5,random_state=42))

        
# train the model
gridCV_object.fit(train_features, train_labels)

print(gridCV_object.best_params_)
print(gridCV_object.score(x_test,y_test))
```

    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 90  92  96  98 121 122 129 134 151 169 183 201 210 216 217 259 261] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 48  49  59  95  96  98  99 134 151 169 183 201 217 228 259 264 268] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:113: UserWarning: Features [ 42  97  98 118 119 123 128 131 134 137 146 151 153 169 201 205 259] are constant.
      UserWarning)
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: invalid value encountered in true_divide
      f = msb / msw
    c:\programdata\anaconda3\lib\site-packages\sklearn\feature_selection\univariate_selection.py:114: RuntimeWarning: divide by zero encountered in true_divide
      f = msb / msw
    

    {'imp__strategy': 'median', 'boost__learning_rate': 0.01, 'imp__missing_values': 'NaN', 'pca__n_components': 64, 'boost__max_depth': 6, 'boost__n_estimators': 4096, 'selection__k': 128}
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-316-277fdee24751> in <module>()
         11 
         12 print(gridCV_object.best_params_)
    ---> 13 print(gridCV_object.score(x_test,y_test))
    

    NameError: name 'x_test' is not defined



```python

```


```python

gridCV_object.best_estimator_.predict(test_features)
```




    array([ 11.64891228,  11.86394811,  12.08985751, ...,  11.86175269,
            11.72694442,  12.20486093])




```python
gridCV_object.best_estimator_
```




    Pipeline(steps=[('imp', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('std', MinMaxScaler(copy=True, feature_range=(0, 1))), ('selection', SelectKBest(k=128, score_func=<function f_classif at 0x000001DE8C409400>)), ('pca', PCA(copy=True, iterated_power='auto', n_components...=4096, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False))])




```python
import pickle
# Save the model if rerun above cell for searching hyperparameters
#with open('model.pkl', 'wb') as source:    
#   s = pickle.dump(gridCV_object.best_estimator_, source)
```


```python
with open('model.pkl','rb') as load:
    model = pickle.load(load)
```


```python
result = model.predict(test_features)
```


```python
np.array(list(zip(testset_index,result))).shape
```




    (1459, 2)




```python
## Output result, match the require format
result_pd= pd.DataFrame(np.array(list(zip(testset_index,result))),
                        columns=['Id','SalePrice']) ## Id start from 1
result_pd.Id = result_pd.Id.astype('int') + 1 ## Id = index + 1
result_pd['SalePrice'] = result_pd['SalePrice'].apply(lambda x: np.e**(x))
```


```python
result_pd[0:5]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>114566.678793</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>142051.948796</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>178056.734844</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>208357.197299</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>183539.943772</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(result_pd).to_csv('result.csv',index = False)
```


```python

```


```python
# train_features_imputed = imputer.fit_transform(train_features) ##Impute Median
# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit_transform(train_features_imputed) ## Min Max Scaling
# train_features[:] = train_features_imputed ## Put the array back into dataframe
```


```python
# train_features.isnull().sum().sum() ## Quick check, Null should be zero
```


```python
# ### Splitting with scikit.learn.train_test_split library
# x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
```


```python
# GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
#                                                min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
# # train_test(GBest, x_train, x_test, y_train, y_test)
```


```python
# GBest.fit(x_train,y_train)
```


```python
# ### Check score for validation test set
# print('Score of Gradienet Boosting Model: ',GBest.score(x_test,y_test))
```

    Score of Gradienet Boosting Model:  0.887710540618
    


```python
# from sklearn.model_selection import GridSearchCV
# ## class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, 
# ## fit_params=None, n_jobs=1, iid=True,refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', 
# ## error_score='raise', return_train_score=True)
```
