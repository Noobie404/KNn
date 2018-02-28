import numpy as np

from sklearn import preprocessing, model_selection, neighbors

import pandas as pd



df = pd.read_csv('DataSet.data.txt')

#we know from breast-cancer-wisconsin.names that 'Missing attribute values: 16'

#need to replace '?' with a nominated value which will not crash and can be filtered out.


df.replace('?', -99999, inplace=True)

#makes the missing data a hugh outlier

df.drop(['SeqNum'], 1, inplace=True)

#drop the id column as it does not aid prediction in any way.



X = np.array(df.drop(['Diabetes'],1))

#drop the 'class' column from X as we will be predicting 'class'

y = np.array(df['Diabetes'])

#use 'class' column for y as this is the column we want to predict.



X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

#split X into train/test data sets.



clf = neighbors.KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)



print ("accuracy=", accuracy)

example_measures = np.array([[8.8,190,550],[8.8,195,550]])
#example_measures = example_measures.reshape(len(example_measures),-1)
#make up some data to predict from when we use the classifier created.
#use len(example_measures) to enable any size of input data to be used.
prediction = clf.predict(example_measures)

print ("prediction=", prediction, "type(prediction)=", type(prediction))
print (df.head())




