```python
import pandas as pd
import numpy as np
data = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\in house 2022\\diabetes.csv")
print(data.head())

array = data.values
X = array[:,0:8]
Y = array[:,8]
```

       Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0            6      148             72             35        0  33.6   
    1            1       85             66             29        0  26.6   
    2            8      183             64              0        0  23.3   
    3            1       89             66             23       94  28.1   
    4            0      137             40             35      168  43.1   
    
       DiabetesPedigreeFunction  Age  Outcome  
    0                     0.627   50        1  
    1                     0.351   31        0  
    2                     0.672   32        1  
    3                     0.167   21        0  
    4                     2.288   33        1  
    


```python
import pandas as pd
import numpy as np
data = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\in house 2022\\diabetes.csv")
print(data)

array = data.values
X = array[:,0:8]
Y = array[:,8]
print("\n filter method implementation \n")
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
                                             
test = SelectKBest(score_func=chi2, k=5)    # Feature extraction
fit = test.fit(X, Y)


np.set_printoptions(precision=3)           # Summarize scores
print(fit.scores_)

features = fit.transform(X)

print(features[0:5,:])                    # Summarize selected features
```

         Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0              6      148             72             35        0  33.6   
    1              1       85             66             29        0  26.6   
    2              8      183             64              0        0  23.3   
    3              1       89             66             23       94  28.1   
    4              0      137             40             35      168  43.1   
    ..           ...      ...            ...            ...      ...   ...   
    763           10      101             76             48      180  32.9   
    764            2      122             70             27        0  36.8   
    765            5      121             72             23      112  26.2   
    766            1      126             60              0        0  30.1   
    767            1       93             70             31        0  30.4   
    
         DiabetesPedigreeFunction  Age  Outcome  
    0                       0.627   50        1  
    1                       0.351   31        0  
    2                       0.672   32        1  
    3                       0.167   21        0  
    4                       2.288   33        1  
    ..                        ...  ...      ...  
    763                     0.171   63        0  
    764                     0.340   27        0  
    765                     0.245   30        0  
    766                     0.349   47        1  
    767                     0.315   23        0  
    
    [768 rows x 9 columns]
    
     filter method implementation 
    
    [ 111.52  1411.887   17.605   53.108 2175.565  127.669    5.393  181.304]
    [[  6.  148.    0.   33.6  50. ]
     [  1.   85.    0.   26.6  31. ]
     [  8.  183.    0.   23.3  32. ]
     [  1.   89.   94.   28.1  21. ]
     [  0.  137.  168.   43.1  33. ]]
    


```python
import pandas as pd
import numpy as np
data = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\in house 2022\\diabetes.csv")


array = data.values
X = array[:,0:8]
Y = array[:,8]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
```

    C:\Users\HP\anaconda3\lib\site-packages\sklearn\utils\validation.py:70: FutureWarning: Pass n_features_to_select=4 as keyword args. From version 1.0 (renaming of 0.25) passing these as positional arguments will result in an error
      warnings.warn(f"Pass {args_msg} as keyword args. From version "
    C:\Users\HP\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Num Features: 4
    Selected Features: [ True  True False False False  True  True False]
    Feature Ranking: [1 1 3 4 5 1 1 2]
    


```python

```


```python

```


```python
import pandas as pd
import numpy as np
data = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\in house 2022\\diabetes.csv")


array = data.values
X = array[:,0:8]
Y = array[:,8]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
```

    C:\Users\HP\anaconda3\lib\site-packages\sklearn\utils\validation.py:70: FutureWarning: Pass n_features_to_select=4 as keyword args. From version 1.0 (renaming of 0.25) passing these as positional arguments will result in an error
      warnings.warn(f"Pass {args_msg} as keyword args. From version "
    C:\Users\HP\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Num Features: 4
    Selected Features: [ True  True False False False  True  True False]
    Feature Ranking: [1 1 3 4 5 1 1 2]
    


```python
import pandas as pd
import numpy as np
data = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\in house 2022\\diabetes.csv")


array = data.values
X = array[:,0:8]
Y = array[:,8]

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X,Y)

Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
print ("Ridge model:", pretty_print_coefs(ridge.coef_))
```

    Ridge model: 0.021 * X0 + 0.006 * X1 + -0.002 * X2 + 0.0 * X3 + -0.0 * X4 + 0.013 * X5 + 0.145 * X6 + 0.003 * X7
    


```python

```
