from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate

iris = datasets.load_iris()
x = iris.data
y = iris.target
X_choose, x_test, y_choose, y_test = train_test_split(x, y, test_size=0.3)

print(X_choose, y_choose)

params = [[0.6652997139930452, 'poly', 7, 4178.386000737241],
          [1.2346990434544882, 'poly', 7, 4317.581190465473],
          [0.8156943235551155, 'poly', 8, 864.1583649816441]]

for c, kernel, degree, gamma in params:
    clf = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
    cv_results = cross_validate(clf, X_choose, y_choose, cv=5,
                                return_train_score=False)
    print(cv_results['test_score'].mean())
