# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:24:42 2022

@author: passi
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import balanced_accuracy_score


random_seed = 42
test_data_size = 0.2
cv_data_size = 0.2
impute_strategy = 'mean'
n_jobs_val = -1 
cv_num = 5
max_iter_num = 200
score_method = 'balanced_accuracy' #balanced_accuracy_score
cv_splitter = ShuffleSplit(n_splits = cv_num, test_size = cv_data_size, random_state = random_seed)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

estimator_names = ['Decision trees', 'Neural networks', 'Boosting', 'Support Vector Machines', 'k-nearest neighbors']
iterative_algorithms = ['Neural networks', 'Support Vector Machines']
estimator_list = [DecisionTreeClassifier, MLPClassifier, GradientBoostingClassifier, SVC, KNeighborsClassifier]
estimators = dict(zip(estimator_names, estimator_list))
    
hyperparameter_list = [{'max_depth': list(range(2, 30, 2)), 'criterion': ['gini', 'entropy']},
    {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)], 'max_iter': list(range(0, 220, 20)),
    'alpha': [0.0001, 0.05], 'learning_rate': ['constant','adaptive'],},
    {'n_estimators': [5,50,250], 'max_depth':[1,3,5], 'learning_rate':[0.01,0.1,1]},
    {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear', 'sigmoid']},
    {'metric': ['euclidean', 'manhattan', 'minkowski'], 'n_neighbors': list(range(2, 30, 2)), 'p': [1,2]}]
hyperparameters = dict(zip(estimator_names, hyperparameter_list))

class supervised_learning:
    def __init__(self, estimator_names):
        self.estimator_names = estimator_names
        self.best_hyperparameters = {}

        
    def get_kepler(self):
        kepler = pd.read_csv("Kepler Exoplanet Search Results.csv", index_col = 'kepid')
        col_to_drop = ['rowid', 'kepoi_name', 'koi_pdisposition', 'koi_tce_delivname']
        col_to_drop.extend(kepler.columns[kepler.isnull().sum()>kepler.shape[0]*0.1].to_list())
        kepler.drop(col_to_drop, axis = 1, inplace = True)
        
        le = LabelEncoder()
        kepler.koi_disposition = le.fit_transform(kepler.koi_disposition)
        y = kepler['koi_disposition']
        print(y.unique(), le.inverse_transform(y.unique()))
        
        X = kepler.loc[:, kepler.columns != 'koi_disposition']
        col_names = X.columns
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        imputer = SimpleImputer(missing_values = np.nan, strategy = impute_strategy)
        imputer = imputer.fit(X)
        X = pd.DataFrame(data = imputer.transform(X), index = y.index, columns = col_names)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size = test_data_size, random_state = random_seed)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
    def tuning_gyperparameter(self, X_train, X_test, y_train, y_test, estimator_name): 
        self.estimator_name = estimator_name
        estimator = estimators[estimator_name]
        parameter_space = hyperparameters[estimator_name]
        clf_func = estimator(random_state = random_seed)
        # higher the AUC value for a classifier, the better its ability to distinguish
        clf = GridSearchCV(clf_func, parameter_space, n_jobs = n_jobs_val, cv = cv_splitter, scoring = score_method, return_train_score = True)
        clf.fit(X_train, y_train)
        print(estimator_name + ' Best parameters found:\n', clf.best_params_)
        self.clf = clf
        params = clf.cv_results_['params']
        train_means = clf.cv_results_['mean_train_score']
        test_means = clf.cv_results_['mean_test_score']
        plt.figure(figsize=(20, 6))
        plt.scatter(list(map(str, params)), train_means, label = 'train_score')
        plt.scatter(list(map(str, params)), test_means, label = 'test_score')
        plt.axvline(x = str(clf.best_params_), ls = '--', label = 'best_param')
        plt.title(self.estimator_name + ' Hyperparameter tuning results', fontsize = 15)
        plt.ylabel('score', fontsize = 12)
        plt.xticks(rotation = 45, ha = 'right', fontsize = 12)
        plt.legend()
        plt.plot()
        
        stds = clf.cv_results_['std_test_score']
        # for train_mean, test_mean, params in zip(train_means, test_means, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        cv_results = pd.concat([pd.DataFrame(params), pd.DataFrame(list(zip(train_means, test_means, stds)), \
                                            columns =['train_score', 'test_score', 'std_test_score'])], axis = 1)
        cv_results.to_csv(estimator_name + '.csv')
            
        self.estimator, self.best_params = estimator, clf.best_params_
        return estimator, clf.best_params_ #estimator(random_state = random_seed, **clf.best_params_)
    
        
    def plot_learning_curve(self, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 5),):
        best_estimator = self.estimator(random_state = random_seed, **self.best_params)
        _, axes = plt.subplots(1, 3, figsize = (20, 5))
    
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            best_estimator, X_train, y_train, cv = cv_splitter, n_jobs = n_jobs_val,
            train_sizes = train_sizes, return_times = True, scoring = score_method)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
    
        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - 2 * train_scores_std,
            train_scores_mean + 2 * train_scores_std, alpha=0.1, color="r",)
        axes[0].fill_between(train_sizes, test_scores_mean - 2 * test_scores_std,
            test_scores_mean + 2 * test_scores_std, alpha=0.1, color="g", )
        axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
        axes[0].legend(loc="best")
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")    
        axes[0].set_title(self.estimator_name + " learning curve")
        
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(train_sizes, fit_times_mean - 2 * fit_times_std,
            fit_times_mean + 2 * fit_times_std, alpha=0.1,)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title(self.estimator_name + " Scalability of the model")
    
        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(fit_time_sorted, test_scores_mean_sorted - 2 * test_scores_std_sorted,
            test_scores_mean_sorted + 2 * test_scores_std_sorted, alpha=0.1, )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title(self.estimator_name + " Performance of the model")
        
        plt.show()
        return
    
    
    def plot_learning_curve_on_iteration(estimator, best_params, X_train, y_train):
        max_iter_num = best_params['max_iter']
        training_score, validation_score = [], []
        
        for train_index, test_index in cv_splitter.split(X_train):
            x_training, x_validation = X_train.iloc[train_index], X_train.iloc[test_index]
            y_training, y_validation = y_train.iloc[train_index], y_train.iloc[test_index]
            training_score_list, validation_score_list = [], []
            for epoch in range(max_iter_num):
                if epoch == 0:
                    best_estimator = estimator(random_state = random_seed, **best_params)
                    best_estimator.warm_start = True
                    best_estimator.max_iter = 1
                best_estimator.partial_fit(x_training, y_training, classes = np.unique(y_training)) 
                y_train_pred = best_estimator.predict(x_training)
                # Multi-layer Perceptron classifier optimizes the log-loss function using LBFGS or stochastic gradient descent.
                curr_train_score = balanced_accuracy_score(y_training, y_train_pred) # training performances
                y_val_pred = best_estimator.predict(x_validation) 
                curr_valid_score = balanced_accuracy_score(y_validation, y_val_pred) # validation performances
                training_score_list.append(curr_train_score) # list of training perf to plot
                validation_score_list.append(curr_valid_score) # list of valid perf to plot
            training_score.append(training_score_list)
            validation_score.append(validation_score_list)
            
        plt.plot(range(max_iter_num), np.array(training_score).mean(axis = 0), label = 'Training')
        plt.plot(range(max_iter_num), np.array(validation_score).mean(axis = 0), label = 'Cross-validation')
        plt.title('Balanced average accuracy under cross validation')
        plt.xlabel('Iterations')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
    
        return training_score, validation_score
        
        
    def predict_test_data(estimator, best_params, X_train, X_test, y_train, y_test):
        best_estimator = estimator(random_state = random_seed, **best_params)
        best_estimator.fit(X_train, y_train)
        y_pred = best_estimator.predict(X_test)
        test_score = balanced_accuracy_score(y_test, y_pred)
        return test_score
    
# sl = supervised_learning(estimator_names)
# X_train, X_test, y_train, y_test = sl.get_kepler()    
# out_of_sample_score = []
# for estimator_name in sl.estimator_names:
#     estimator, best_params = sl.tuning_gyperparameter(X_train, X_test, y_train, y_test, estimator_name)
#     sl.best_hyperparameters[estimator_name] = best_params
#     sl.plot_learning_curve(X_train, y_train)
#     if estimator_name in iterative_algorithms:
#         training_score, validation_score = sl.plot_learning_curve_on_iteration(estimator, best_params, X_train, y_train)
#     test_score = sl.predict_test_data(estimator, best_params, X_train, X_test, y_train, y_test)
#     out_of_sample_score.append(test_score)
    
    
sl = supervised_learning(estimator_names)
estimator_name = 'Boosting'
X_train, X_test, y_train, y_test = sl.get_kepler() 
estimator, best_params = sl.tuning_gyperparameter(X_train, X_test, y_train, y_test, estimator_name)
# best_estimator = estimator(random_state = random_seed, **best_params)
# _ = plot_learning_curve(best_estimator, X_train, y_train)
# best_estimator = estimator(random_state = random_seed, **best_params)
# plot_learning_curve_on_iteration(estimator, best_params, X_train, y_train)



