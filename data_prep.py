import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import random

# Load the dataset
data = pd.read_excel("/content/Brain_Dat.xlsx")  # Replace with your dataset path
data=data.drop(data.columns[0],axis=1)

# Separate features and labels
y = data.iloc[:, 0].values  # The first column contains the labels
X = data.iloc[:, 1:].values  # All other columns are features

# Encode labels if necessary
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
