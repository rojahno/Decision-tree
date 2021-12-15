# Decision Tree


```python
%load_ext autoreload
```


```python
%autoreload

import numpy as np 
import pandas as pd 
import decision_tree as dt  # <-- Your implementation
```

## First Dataset


```python
data_1 = pd.read_csv('data_1.csv')
data_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Wind</th>
      <th>Play Tennis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>Weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Overcast</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sunny</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Overcast</td>
      <td>Mild</td>
      <td>High</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



### Fit and Evaluate Model


```python
# Separate independent (X) and dependent (y) variables
X = data_1.drop(columns=['Play Tennis'])
y = data_1['Play Tennis']

# Create and fit a Decrision Tree classifier
model_1 = dt.DecisionTree()  # <-- Should work with default constructor
model_1.fit(X,y)

# Verify that it perfectly fits the training set
print(f'Accuracy: {dt.accuracy(y_true=y, y_pred=model_1.predict(X)) * 100 :.1f}%')
```

    Accuracy: 100.0%
    

### Inspect Classification Rules

A big advantage of Decision Trees is that they are relatively transparent learners. By this we mean that it is easy for an outside observer to analyse and understand how the model makes its decisions. The problem of being able to reason about how a machine learning model reasons is known as _Explainable AI_ and is often a desirable property of machine learning systems.


```python
model_1.print_rules("Yes")
```

    ❌ Outlook=Sunny ∩ Humidity=High => No
    ✅ Outlook=Sunny ∩ Humidity=Normal => Yes
    ✅ Outlook=Overcast => Yes
    ✅ Outlook=Rain ∩ Wind=Weak => Yes
    ❌ Outlook=Rain ∩ Wind=Strong => No
    

## Second Dataset


```python
data_2 = pd.read_csv('data_2.csv')
data_2 = data_2.drop(columns=['Founder Zodiac']) # Drops the column that creates noise in the learning
data_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Founder Experience</th>
      <th>Second Opinion</th>
      <th>Competitive Advantage</th>
      <th>Lucurative Market</th>
      <th>Outcome</th>
      <th>Split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>moderate</td>
      <td>negative</td>
      <td>yes</td>
      <td>no</td>
      <td>success</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>positive</td>
      <td>yes</td>
      <td>no</td>
      <td>failure</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>negative</td>
      <td>no</td>
      <td>no</td>
      <td>failure</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>low</td>
      <td>negative</td>
      <td>no</td>
      <td>no</td>
      <td>failure</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>low</td>
      <td>positive</td>
      <td>yes</td>
      <td>yes</td>
      <td>success</td>
      <td>train</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>moderate</td>
      <td>positive</td>
      <td>no</td>
      <td>yes</td>
      <td>failure</td>
      <td>test</td>
    </tr>
    <tr>
      <th>196</th>
      <td>low</td>
      <td>negative</td>
      <td>no</td>
      <td>yes</td>
      <td>failure</td>
      <td>test</td>
    </tr>
    <tr>
      <th>197</th>
      <td>moderate</td>
      <td>negative</td>
      <td>no</td>
      <td>yes</td>
      <td>failure</td>
      <td>test</td>
    </tr>
    <tr>
      <th>198</th>
      <td>moderate</td>
      <td>negative</td>
      <td>no</td>
      <td>no</td>
      <td>failure</td>
      <td>test</td>
    </tr>
    <tr>
      <th>199</th>
      <td>moderate</td>
      <td>negative</td>
      <td>yes</td>
      <td>no</td>
      <td>success</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 6 columns</p>
</div>



### Split Data

The data is split into three sets:

- `train` contains 50 samples that you should use to generate the tree
- `valid` contains 50 samples that you can use to evaluate different preprocessing methods and variations to the tree-learning algorithm.
- `test` contains 100 samples and should only be used to evaluate the final model once you're done experimenting.


```python
data_2_train = data_2.query('Split == "train"')
data_2_valid = data_2.query('Split == "valid"')
data_2_test = data_2.query('Split == "test"')
X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome
data_2.Split.value_counts()
```




    test     100
    train     50
    valid     50
    Name: Split, dtype: int64



### Fit and Evaluate Model


```python
# Fit model
model_2 = dt.DecisionTree()  # <-- Feel free to add hyperparameters 
model_2.fit(X_train, y_train)
print(f'Train: {dt.accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
print(f'Valid: {dt.accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')
```

    Train: 92.0%
    Valid: 88.0%
    

### Inspect Classification Rules


```python
model_2.print_rules(outcome="success")
```

    ✅ Founder Experience=moderate ∩ Competitive Advantage=yes ∩ Lucurative Market=no ∩ Second Opinion=success => success
    ❌ Founder Experience=moderate ∩ Competitive Advantage=yes ∩ Lucurative Market=yes => failure
    ❌ Founder Experience=moderate ∩ Competitive Advantage=no ∩ Second Opinion=positive ∩ Lucurative Market=failure => failure
    ❌ Founder Experience=moderate ∩ Competitive Advantage=no ∩ Second Opinion=negative => failure
    ❌ Founder Experience=high ∩ Lucurative Market=no => failure
    ✅ Founder Experience=high ∩ Lucurative Market=yes ∩ Competitive Advantage=no ∩ Second Opinion=success => success
    ❌ Founder Experience=high ∩ Lucurative Market=yes ∩ Competitive Advantage=yes => failure
    ❌ Founder Experience=low ∩ Second Opinion=negative => failure
    ✅ Founder Experience=low ∩ Second Opinion=positive => success
    
