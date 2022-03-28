from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import random


## REPLACE WITH ML_TOOLS FUNCTION
def categorical_encode(values):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)


def one_hot_acc(y_true, y_pred):
    out = [np.array_equal(t, p) for t, p in zip(y_true, y_pred)]
    return sum(out)/len(out)
    


dtc = DecisionTreeClassifier(max_depth=2, random_state=42)



# training_loop()
df = pd.read_excel('data/classified/classified_processed_Harm_Jordan_Walking.xlsx')

# Get features and labels
X = df.drop(['classification', 'Time_1'], axis=1).to_numpy()
y = categorical_encode(df['classification'])

# Split data into train, test, val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
# X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=False)

dtc.fit(X_train, y_train)

print('Train accuracy:', dtc.score(X_train, y_train))
print('Val accuracy:', dtc.score(X_val, y_val))
# print('Test accuracy: {0}'.format(dtc.score(X_test, y_test)))