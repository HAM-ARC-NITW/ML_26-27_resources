from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

cancer= load_breast_cancer()
x=cancer.data[:,:2]
y=cancer.target
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.2, random_state =7)
svm=SVC(kernel='linear', C=1)
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

fig, ax= plt.subplots(figsize=(10,8))
DecisionBoundaryDisplay.from_estimator( svm , x_train, 
                                       response_method = "predict", 
                                       alpha=.8, 
                                       cmap="Pastel1",
                                       ax=ax,
                                       xlabel=cancer.feature_names[0],
                                       ylabel=cancer.feature_names[1])
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, s=30, edgecolors='k')
plt.title('Breast Cancer Dataset')
plt.savefig('breast_cancer_linear_kernel.png', dpi=300)
