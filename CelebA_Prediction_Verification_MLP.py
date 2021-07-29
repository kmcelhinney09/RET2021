import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

celebA_data_path = 'CelebA_prediction_numpy'
celebA_labels_path = 'CelebA_predictions_numpy_labels'

with open(celebA_data_path, 'rb') as data:
    celebA_data = pickle.load(data)

with open(celebA_labels_path, 'rb') as labels:
    celebA_labels = pickle.load(labels)

# print(celebA_data.shape)
# print(celebA_labels.shape)

x_train, x_test, y_train, y_test = train_test_split(
    celebA_data,
    celebA_labels,
    test_size=.2,
    random_state=1
)
# create an MLPClassifier to put image data through and fit it
clf = MLPClassifier(solver='sgd',
                    activation='relu',
                    alpha=1,
                    hidden_layer_sizes=(512, 512),
                    # early_stopping=True,
                    # tol=1e-5,
                    # validation_fraction=.2,
                    verbose=True
                    )

clf.fit(x_train, y_train)

# Find accuracy for output file for later review
score = clf.score(x_test, y_test)
print(score)
