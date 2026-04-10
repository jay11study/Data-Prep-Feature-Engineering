import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

# Create dataset
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'City': ['Mumbai', 'Delhi', 'Chennai', 'Mumbai', 'Delhi'],
    'Size': ['Small', 'Large', 'Medium', 'Large', 'Small'],
    'Age': [25, 45, 32, 60, 28],
    'Fare': [15, 300, 85, 450, 20]
}


df = pd.DataFrame(data)

#Task 1 — Categorical Encoding

# Label Encoding for Gender
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Label Encoding for Size with order
size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
df['Size'] = df['Size'].map(size_mapping)

# One-Hot Encoding for City
df = pd.get_dummies(df, columns=['City'], drop_first=True)

#Task 2 — Feature Scaling

# Apply RobustScaler because Fare has an outlier (450)
scaler = RobustScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Final Output
print(df)

#---------------------------------------------------------------------------
# Justification for the scaler choice:
# RobustScaler is used instead of StandardScaler or MinMaxScaler because the Fare column contains an outlier (450).
# RobustScaler uses median and interquartile range (IQR), making it less sensitive to extreme values.


