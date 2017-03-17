import numpy as np

f = open("new_train.csv")
data =  np.loadtxt(fname = f, delimiter = ',')
label = data[:, 1]
pixels = data[:, 9:438]

g = open("test.csv")
test_data =  np.loadtxt(fname = g, delimiter = ',')
test_pixels = test_data[:, 9:438]
test_label = test_data[:, 1]

from sklearn.neighbors import KNeighborsClassifier

#Create the knn model.
#Look at the five closest neighbors.
knn = KNeighborsClassifier(n_neighbors=5)

#Fit the model on the training data.
knn.fit(pixels, label)

#Make point predictions on the test set using the fit model.
predictions = knn.predict(test_pixels)

print (predictions)
test_id = map(int, test_data[:, 0])
np.savetxt("submission.csv", np.column_stack((test_id, predictions)), delimiter=",", fmt='%s', header="Id,Character")
