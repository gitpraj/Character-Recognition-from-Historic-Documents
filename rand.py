import numpy as np

f = open("train.csv")
  
test_data = np.loadtxt(fname = f, delimiter = ',')

Y_train = test_data[:, 1].astype("int")

X_train = test_data[:, 9:438]
g = open("test.csv")
testing =  np.loadtxt(fname = g, delimiter = ',')

X_test = testing[:, 9:438]

Y_test = testing[:, 1].astype("int")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, Y_train)
clf.predict(X_test)
test_id =  testing[:, 0].astype("int")
classes = clf.predict(X_test)
print (test_id)
np.savetxt("submission1.csv", np.column_stack((test_id, classes)), delimiter=",", fmt='%s', header="Id,Character")
