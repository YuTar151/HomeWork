import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import hog
from skimage import color
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_data_gray = np.array([cv2.cvtColor(x.reshape(32, 32, 3), cv2.COLOR_BGR2GRAY) for x in x_train])
test_data_gray = np.array([cv2.cvtColor(x.reshape(32, 32, 3), cv2.COLOR_BGR2GRAY) for x in x_test])

train_hog_features = np.array([hog(x, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) for x in train_data_gray])
test_hog_features = np.array([hog(x, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) for x in test_data_gray])

pca = PCA(n_components=200, random_state=42)
train_pca_features = pca.fit_transform(train_hog_features)
test_pca_features = pca.transform(test_hog_features)

y_train = y_train.ravel()
y_test = y_test.ravel()

clf = SVC(kernel='rbf')
clf.fit(train_pca_features, y_train)

y_pred = clf.predict(test_pca_features)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', rec)
print('F1-Score:', f1)
