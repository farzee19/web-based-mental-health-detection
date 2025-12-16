from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

# import dataset
data_set = pd.read_csv ('mental health-cleaned.csv')
# split data into X and y
data_set.drop(columns='Unnamed: 0', inplace=True)
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 10].values

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=0.20, random_state=1)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)


#Fitting XGBoost classifier to the training set  
classifier = XGBClassifier(
    base_score=0.5, booster='gbtree', colsample_bylevel=1,
    colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
    gamma=0, importance_type='gain',
    learning_rate=0.300000012,
    max_delta_step=0, max_depth=6, min_child_weight=1,
    n_estimators=100, n_jobs=16,
    objective='multi:softmax', num_class=5, random_state=0,
    reg_alpha=0, reg_lambda=1, subsample=1,
    tree_method='exact', use_label_encoder=False
)

# 1. Fit the model
print("Training model...")
classifier.fit(X_train, y_train)

# 2. SAVE THE ACTUAL CLASSIFIER OBJECT
joblib.dump(classifier, 'model.pkl')
joblib.dump(label_encoder, 'encoder.pkl')

print("Model and Encoder saved successfully!")

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
