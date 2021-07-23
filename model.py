import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics


from warnings import filterwarnings
filterwarnings(action='ignore')

thyroid_data = pd.read_csv("thyroid_data.csv")
thyroid_data
thyroid_data = thyroid_data.drop(['S.no'], axis = 1)

# A quick fix needed
thyroid_data.loc[thyroid_data['Age'] == '455', 'Age'] = '45'


## Let's drop some unnecessary columns
thyroid_data = thyroid_data.drop(['TSH Measured','T3 Measured','TT4 Measured','T4U Measured','FTI Measured',],axis=1)

#Checking for null values
thyroid_data.isna().sum()

from sklearn.linear_model import LogisticRegression



def convert_category(dataframe, column):

    if column == 'Sex':
        conditionF = dataframe[column] == 'F' # For sex column
        conditionT = dataframe[column] == 'M' # For sex column
    else:
        conditionF = dataframe[column] == 'f'
        conditionT = dataframe[column] == 't'

    dataframe.loc[conditionF, column] = 0
    dataframe.loc[conditionT, column] = 1

# Binarize Category Columns
binary_cols = ['Age', 'Sex', 'On Thyroxine', 'Query on Thyroxine',
       'On Antithyroid Medication', 'Sick', 'Pregnant', 'Thyroid Surgery',
       'I131 Treatment', 'Query Hypothyroid', 'Query Hyperthyroid', 'Lithium',
       'Goitre', 'Tumor', 'Hypopituitary', 'Psych', 'TSH', 'T3', 'TT4', 'T4U',
       'FTI']

for col in binary_cols: convert_category(thyroid_data, col)


# Convert '?' to np.nan and convert numeric data to numeric dtype
for col in thyroid_data.columns:
    if col != 'Category':
        thyroid_data.loc[thyroid_data[col] == '?', col] = np.nan
        thyroid_data[col] = pd.to_numeric(thyroid_data[col])

from sklearn.impute import SimpleImputer

curr_columns = thyroid_data.columns.difference(['Category'])

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data = imputer.fit_transform(thyroid_data.drop('Category', axis=1))
imputed_data = pd.DataFrame(imputed_data, columns=curr_columns)


thyroid_data = pd.concat([
                    imputed_data.reset_index(),
                    thyroid_data['Category'].reset_index()],
                    axis=1).drop('index', axis=1)


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = thyroid_data.drop('Category', axis=1)
y = thyroid_data['Category']

col_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

#Using LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print('Accuracy:',metrics.accuracy_score(prediction,y_test))

# Saving model to disk
pickle.dump(model, open('thyroid_model.pkl','wb'))

# Loading model to compare the results
thyroid_model = pickle.load(open('thyroid_model.pkl','rb'))
print(model.predict([[20,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.5,2.5,200,2,200]]))
