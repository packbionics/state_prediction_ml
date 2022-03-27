from matplotlib.pyplot import ylabel
import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def training_loop(X):
    # Model params
    batch_size = 32  # ~3333 rows per dataset
    num_features = X.shape[0]
    
    # Training params
    epochs = 100
    learning_rate = 0.001
    # weight_decay = 1e-8

## REPLACE WITH ML_TOOLS FUNCTION
def categorical_encode(values):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)
    

if __name__ == '__main__':
    
    # training_loop()
    df = pd.read_excel('data/classified/classified_processed_Harm_Jordan_Walking.xlsx')
    
    # Get features and labels
    X = df.drop(['classification', 'Time_1'], axis=1).to_numpy()
    y = categorical_encode(df['classification'])
    
    
    
