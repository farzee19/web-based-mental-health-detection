# libraries pre processing
import pandas as pd # used for handling the dataset
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.compose import ColumnTransformer 
import matplotlib.pyplot as plt

data_set = pd.read_csv('mental health (original).csv') # to import the dataset into a variable
# Splitting the attributes into independent and dependent attributes
X = data_set.iloc[:, :-1] # attributes to determine dependent variable / Class
y = data_set.iloc[:, 22] # dependent variable / Class

#checking missing value
displaymissing=data_set.isnull().sum()
print(displaymissing)

# Iterate over the index range from
# 0 to max number of columns in dataframe
for index in range(data_set.shape[1]):    
    # Select column by index position using iloc[]
    columnSeriesObj = data_set.iloc[:, index]
    columnSeriesObj[columnSeriesObj =='Yes']=1
    columnSeriesObj[columnSeriesObj =='No']=0


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

from sklearn.feature_selection import mutual_info_classif, SelectKBest
#determine the mutual information
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info
mutual_info = pd.Series(mutual_info) 
mutual_info.index = X.columns
print(mutual_info.sort_values(ascending=False))

mutual_info.sort_values(ascending=False).plot.bar(figsize=(25, 10))
plt.show()

#select the top 10 features
sel_ten_cols = SelectKBest(mutual_info_classif, k=10)
sel_ten_cols.fit(X_train, y_train)
X_train.columns[sel_ten_cols.get_support()]

cols = sel_ten_cols.get_support()
selected_columns = X.iloc[:,cols].columns.tolist()
print(selected_columns)

data_set = data_set.drop(columns=[X for X in data_set if X not in selected_columns])
data_set['Disorder'] = y
print(data_set)

#write dataframe to csv file
data_set.to_csv('mental health-cleaned.csv') 