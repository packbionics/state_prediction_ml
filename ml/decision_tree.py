from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


## REPLACE WITH ML_TOOLS FUNCTION
def categorical_encode(values):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)


dtc = DecisionTreeClassifier()


params_dict = {
 'criterion': ['gini', 'entropy'],
 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#  'max_features': [None, 'auto', 'sqrt', 'log2'],
#  'random_state': [None, 42],
#  'max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#  'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#  'class_weight': [None, 'balanced'],
#  'cpp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}



# training_loop()
df = pd.read_excel('data/classified/classified_processed_Harm_Jordan_Walking.xlsx')

# Get features and labels
X = df.drop(['classification', 'Time_1'], axis=1).to_numpy()
y = categorical_encode(df['classification'])

# Split data into train, test, val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)
# X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=False)

grid_search = GridSearchCV(dtc, params_dict, cv=5)
grid_search.fit(X_train, y_train)
tuned_dtc = grid_search.best_estimator_
print(tuned_dtc)

print('Train accuracy:', tuned_dtc.score(X_train, y_train))
print('Val accuracy:', tuned_dtc.score(X_val, y_val))
print(cross_val_score(tuned_dtc, X, y, cv=5))
# print('Test accuracy: {0}'.format(dtc.score(X_test, y_test)))