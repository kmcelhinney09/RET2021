from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def accuracy(cm):
    diagonal = cm.trace()
    elem=cm.sum()
    return diagonal/elem

digits = load_digits()
#training
x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

clf = MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(64,64))

clf.fit(x_train,y_train)

prediction = clf.predict(x_test)
acc = confusion_matrix(y_test, prediction)

print(accuracy(acc))



