from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

def stitch(ds):
	row, col = len(ds[0]), len(ds)
	data = np.zeros((row, col), dtype=int)
	print(np.shape(data))
	for i in range(col):
		for j in range(row):
			data[j][i] = ds[i][j]
	return data

path = "Resources/Diabetes.csv"
dataset = pd.read_csv(path)

# creating labelEncoder
encoder = preprocessing.LabelEncoder()
encoded_dataset = []
attributes = dataset.columns
for attr in attributes:
    encoded_dataset.append(encoder.fit_transform(list(dataset[attr])))
features = stitch(encoded_dataset)
test = []
test.append(features[-2])
test.append(features[-1])
features = features[:-2,:]
label = encoder.fit_transform(dataset[attributes[-1]].iloc[:-2])

# Creating Naive-Bayes-Classifier model.
naive_bayes = GaussianNB()

# training the model
naive_bayes.fit(features, label)

#predicting using naive-bayes.
predicted = naive_bayes.predict([test[0]])
print(predicted)

predicted = naive_bayes.predict([test[1]])
print(predicted)

