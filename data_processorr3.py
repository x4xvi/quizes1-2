import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("Darknet.csv")
# Drop unwanted columns
df = df.drop(["Flow ID", "Timestamp", "Label2", "Src IP", "Dst IP"], axis=1)
# Drop rows with missing values
df = df.dropna()

# Convert columns to integer type
for col in ['Src Port', 'Dst Port', 'Protocol']:
    df[col] = df[col].astype(int)

# Encode categorical labels
label_encoder = LabelEncoder()
df['Label1'] = label_encoder.fit_transform(df['Label1'])

# Save the processed data
df.to_csv("processed.csv", index=False)

# Scaling features with RobustScaler
scaler = RobustScaler()

# Function to scale data, skipping large or infinite values
def robust_scale_skip_large(X):
    with np.errstate(over='ignore'):
        try:
            return scaler.fit_transform(X)
        except ValueError as e:
            print(f"Skipping scaling: {str(e)}")
            return X  # Return original data if scaling fails

# Apply scaling
scaled_features = robust_scale_skip_large(df.drop('Label1', axis=1))

# Create DataFrame for scaled features
scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])
scaled_df['Label1'] = df['Label1']

# Save the scaled dataset
scaled_df.to_csv("scaled3.csv", index=False)

# Load scaled data
scaled_df = pd.read_csv("scaled3.csv")
