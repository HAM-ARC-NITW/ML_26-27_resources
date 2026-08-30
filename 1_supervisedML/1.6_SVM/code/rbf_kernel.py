from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris= load_iris()
x=iris.data
y=iris.target
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=7)
svm= SVC(kernel='rbf', C=1, gamma="scale")
svm.fit(x_train, y_train)
y_pred=svm.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))
