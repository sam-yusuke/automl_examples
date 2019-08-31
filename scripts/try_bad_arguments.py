from sklearn import datasets
import hyperopt.pyll.stochastic
from hyperopt import tpe, hp, fmin
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import tqdm

iris = datasets.load_iris()
x = iris.data
y = iris.target
X_choose,x_test,y_choose,y_test = train_test_split(x,y,test_size=0.3)

def create_classifier(args):
    C = args['param']['C']
    kernel = args['param']['kernel']
    degree = args['param']['degree']
    gamma = args['param']['gamma']
    print(C, kernel, degree, gamma)
    clf = SVC(C=C, kernel=kernel, degree=degree,gamma=gamma)

    return clf

def objective_func(args):
    clf = create_classifier(args)
    cv_results = cross_validate(clf, X_choose, y_choose, cv=5,
                                return_train_score=False)
    return -cv_results['test_score'].mean()

def main():
    space = hp.choice('classifier',[
            {'model': SVC,
            'param':{'C':hp.lognormal('C',0,1),
            'kernel':hp.choice('kernel',['rbf','rbf','sigmoid']),
            'degree':hp.choice('degree',range(1,15)),
            'gamma':hp.uniform('gamma',0.001,10000)}
            }
            ])

    for _ in tqdm.tqdm(range(0, 10000)):
        sample = hyperopt.pyll.stochastic.sample(space)
        objective_func(sample)

if __name__ == "__main__":
    main()