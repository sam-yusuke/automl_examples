#%%
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

#%%
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

#%%
import json
obj = json.load(open("scripts/test.json"))
X, y = obj['X'], obj['y']

#%%
import numpy as np
non_infinity_dataset = list(zip(*[(x, y) for x, y in zip(X, y) if all(map(lambda x: not np.isinf(x), x))]))
X, y = np.array(non_infinity_dataset[0]), np.array(non_infinity_dataset[1])

#%%
import numpy as np

print(X.shape)
print(y.shape)

#%%
indices = list(range(0, len(X)))
np.random.shuffle(indices)
random_100 = indices[:100]
X_100 = X[random_100]
y_100 = y[random_100]

# print(X_100, y_100)

#%%
# X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

print(X_test.shape)
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
    output_folder='/tmp/autosklearn_parallel_1_example_out',
    disable_evaluator_output=False,
    # 'holdout' with 'train_size'=0.67 is the default argument setting
    # for AutoSklearnClassifier. It is explicitly specified in this example
    # for demonstrational purpose.
    resampling_strategy='holdout',
    resampling_strategy_arguments={'train_size': 0.67},
    n_jobs=2,
    seed=3,
    delete_output_folder_after_terminate=False,
    delete_tmp_folder_after_terminate=False,
)
# automl.fit(X_train, y_train, dataset_name='breast_cancer')
automl.fit(X_train, y_train)

# Print the final ensemble constructed by auto-sklearn.
print(automl.show_models())
predictions = automl.predict(X_test)
# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out.
print(automl.sprint_statistics())
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
print(sklearn.metrics.confusion_matrix(y_test, predictions))

