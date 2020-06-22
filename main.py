import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import sklearn 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score



#reading in data
colnames = [ 'Race' , 'Age' , 'Trial', 'LettersSent' , 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Test']
df = pd.read_csv('mod - Copy.csv', names=colnames, usecols=['Race', 'Age', 'Trial', 'LettersSent', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Test'])


#running data through encoder
preprocessor = make_column_transformer( (OneHotEncoder(),[0,1,2,3,4,5,6,7,8]),remainder="passthrough")
#dropping result column
df.drop(['Test'], axis=1)

#replacing null with 0
df = df.fillna(0)



#declaring and storing logistic regression model
lm = linear_model.LogisticRegression(solver='lbfgs')

#feeding through pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                  ('classifier', lm)])

#declaring x and y variables 
x = df[colnames[:-1]]
y = df[colnames[-1]]
x = preprocessor.fit_transform(x)
#Verify process is done correctly
print(x)
#declaring training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42,test_size = 0.2)

#making it dense in order to make to fit it
X_train.toarray()
#fitting training data through model
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
# accuracy of model being evaluated and printed
accuracy = accuracy_score(y_test, predictions)
plt.show()
print(accuracy) 
print(predictions)