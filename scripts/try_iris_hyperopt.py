from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import tpe, hp, fmin

iris = datasets.load_iris()
x = iris.data
y = iris.target
X_choose,x_test,y_choose,y_test = train_test_split(x,y,test_size=0.3)

def create_classifier(args):
    if args['model']==KNeighborsClassifier:
        n_neighbors = args['param']['n_neighbors']
        algorithm = args['param']['algorithm']
        leaf_size = args['param']['leaf_size']
        metric = args['param']['metric']
        clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               metric=metric,
                               )
    elif args['model']==SVC:
        C = args['param']['C']
        kernel = args['param']['kernel']
        degree = args['param']['degree']
        gamma = args['param']['gamma']
        clf = SVC(C=C, kernel=kernel, degree=degree,gamma=gamma)

    return clf

def objective_func(args):
    clf = create_classifier(args)
    scores = cross_val_score(clf, X_choose, y_choose, cv=5)
    print("Validation Score:",scores.mean())
    print("\n=================")
    return -scores.mean()

space = hp.choice('classifier',[
        {'model': KNeighborsClassifier,
        'param': {'n_neighbors':
                        hp.choice('n_neighbors',range(3,11)),
        'algorithm':hp.choice('algorithm',['ball_tree','kd_tree']),
        'leaf_size':hp.choice('leaf_size',range(1,50)),
        'metric':hp.choice('metric', ["euclidean","manhattan",
                           "chebyshev","minkowski"
                           ])}
        },
        {'model': SVC,
        'param':{'C':hp.lognormal('C',0,1),
        'kernel':hp.choice('kernel',['rbf','poly','rbf','sigmoid']),
        'degree':hp.choice('degree',range(1,15)),
        'gamma':hp.uniform('gamma',0.001,10000)}
        }
        ])

from hyperopt import fmin, tpe, space_eval
from sklearn.metrics import confusion_matrix
best_classifier = fmin(objective_func,space,
                        algo=tpe.suggest,max_evals=100)
print(space_eval(space, best_classifier))
clf = create_classifier(space_eval(space, best_classifier))
clf.fit(X_choose, y_choose)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))