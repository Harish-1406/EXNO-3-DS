## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
df
df1=df.copy()
climate=['Warm','Cold','Hot','Very Hot']
E1=OrdinalEncoder(categories=[climate])
E1.fit_transform(df1[["Ord_1"]])
```

<img width="231" height="213" alt="image" src="https://github.com/user-attachments/assets/fa524fbb-c8bf-4020-98ef-3e682d3bf2b0" />

```py
df1['bo2']=E1.fit_transform(df1[["Ord_1"]])
df1
```
<img width="565" height="374" alt="image" src="https://github.com/user-attachments/assets/84214086-7023-42ea-b90c-3a9ddc1a8b85" />

```py
## Label Encoder
le=LabelEncoder()
df2=df.copy()
df2['Ord_2']=le.fit_transform(df2['Ord_2'])
df2
```

<img width="455" height="372" alt="image" src="https://github.com/user-attachments/assets/949c7553-9a7a-476c-81fa-79b79887507b" />

```py
## One hot Encoder
df3=df.copy()
ohe=OneHotEncoder()
new_data=pd.DataFrame(ohe.fit_transform(df3[['City']]))
df4=pd.concat([df3,new_data],axis=1)
df4
```

<img width="719" height="372" alt="image" src="https://github.com/user-attachments/assets/01e9c744-3c0f-4135-897c-4058c0a9b4d7" />

```py
pd.get_dummies(df4,columns=['City'])
```

<img width="1049" height="391" alt="image" src="https://github.com/user-attachments/assets/e724e390-40ed-42ca-b1cd-84340bdbe97e" />

```py
from category_encoders import BinaryEncoder
df5=df.copy()
be=BinaryEncoder()
new_data=pd.DataFrame(be.fit_transform(df5[['Ord_1']]))
df6=pd.concat([df5,new_data],axis=1)
df6
```

<img width="716" height="380" alt="image" src="https://github.com/user-attachments/assets/d49cbe46-5e4b-429b-b1f7-7390ebd825d0" />

```py
## Target Encoder
from category_encoders import TargetEncoder
df7=df.copy()
te=TargetEncoder()
new_data=pd.DataFrame(te.fit_transform(df7[['Ord_1']],df7['Target']))
df8=pd.concat([df7,new_data],axis=1)
df8
```

<img width="610" height="373" alt="image" src="https://github.com/user-attachments/assets/2ac3ee0c-94a5-4f32-926b-930c8e945288" />

```py
df=pd.read_csv('Data_to_Transform.csv')
df
```

<img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/a207f8bc-d17c-4634-b9c0-fae1e55cbf46" />

```py
df.skew()
```

<img width="368" height="126" alt="image" src="https://github.com/user-attachments/assets/8e4ac126-d428-48db-ba2a-006a309edd51" />

```py
np.log(df["Highly Positive Skew"])
```

<img width="561" height="270" alt="image" src="https://github.com/user-attachments/assets/b45c5d60-53b8-4fa9-948d-c7efeb842549" />

```py
np.reciprocal(df["Moderate Negative Skew"])
```

<img width="585" height="269" alt="image" src="https://github.com/user-attachments/assets/fcbf7bd3-cd6e-46b5-9ac6-f2d9baa40c2e" />

```py
np.sqrt(df["Highly Positive Skew"])
```

<img width="591" height="262" alt="image" src="https://github.com/user-attachments/assets/dcfbc990-8415-4da8-b9bf-94dfde3279c0" />

```py
np.square(df["Highly Positive Skew"])
```

<img width="564" height="256" alt="image" src="https://github.com/user-attachments/assets/7136b702-3556-4644-8b02-0693cdc10f7b" />

```py
from scipy import stats
df["Highly Positive Skew"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="778" height="451" alt="image" src="https://github.com/user-attachments/assets/ce86509b-9472-4b5b-beb5-405b1fbe05f0" />

```py
df["Moderate Negative Skew"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```

<img width="801" height="448" alt="image" src="https://github.com/user-attachments/assets/06b32970-27b2-48cc-8280-afcbce124734" />

```py
df.skew()
```

<img width="345" height="110" alt="image" src="https://github.com/user-attachments/assets/0a801bc6-13b2-4f76-925e-78561d0d0772" />

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="435" height="136" alt="image" src="https://github.com/user-attachments/assets/e64ab185-23ba-42eb-b463-f2d223bd86e4" />

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1255" height="465" alt="image" src="https://github.com/user-attachments/assets/350e3bc6-018a-473c-af6f-4f3c87d54d38" />

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="754" height="549" alt="image" src="https://github.com/user-attachments/assets/5d9e7bc4-393d-4ca5-8070-5a924bae7f82" />

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="757" height="551" alt="image" src="https://github.com/user-attachments/assets/2f257988-7add-4522-abf5-898d82a5eca8" />

```py
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="725" height="547" alt="image" src="https://github.com/user-attachments/assets/8e723607-cc92-4a9a-8084-92e273efc10e" />

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="718" height="534" alt="image" src="https://github.com/user-attachments/assets/fbc03fd2-e842-4d62-89ee-a2e656c2b5ad" />

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="741" height="528" alt="image" src="https://github.com/user-attachments/assets/ffbde4a4-77f4-4969-badd-b0560de56bc4" />

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
