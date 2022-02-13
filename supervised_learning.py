# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:24:42 2022

@author: passi
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, GridSearchCV, learning_curve
from sklearn.metrics import balanced_accuracy_score
# from sklearn.pipeline import make_pipeline

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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
    {'hidden_layer_sizes': [(50,50,50), (100,)], 'max_iter': [50, 100, 200, 500],
    'alpha': [0.0001, 0.05], 'learning_rate': ['constant','adaptive'],},
    {'n_estimators': [5,50,250], 'max_depth':[1,3,5], 'learning_rate':[0.01,0.1,1]},
    {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'linear'], 'max_iter': [50, 200, 500, 1000, 2000]},
    {'n_neighbors': list(range(5, 35, 5)), 'p': [1, 2, 3]}]
hyperparameters = dict(zip(estimator_names, hyperparameter_list))

class supervised_learning:
    def __init__(self, estimator_names):
        self.estimator_names = estimator_names
        self.best_hyperparameters = {}

            
    def data_preprocess(self, X, y):
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
    
    
    def get_kepler_data(self):
        kepler = pd.read_csv("Kepler Exoplanet Search Results.csv", index_col = 'kepid')
        col_to_drop = ['rowid', 'kepoi_name', 'koi_pdisposition', 'koi_tce_delivname']
        col_to_drop.extend(kepler.columns[kepler.isnull().sum()>kepler.shape[0]*0.1].to_list())
        kepler.drop(col_to_drop, axis = 1, inplace = True)
        
        le = LabelEncoder()
        kepler.koi_disposition = le.fit_transform(kepler.koi_disposition)
        y = kepler['koi_disposition']
        print(y.unique(), le.inverse_transform(y.unique()))
        
        X = kepler.loc[:, kepler.columns != 'koi_disposition']
        return self.data_preprocess(X, y)
    
    
    def get_maternal_data(self):
        maternal = pd.read_csv('Maternal Health Risk Data Set.csv')
        le = LabelEncoder()
        maternal.RiskLevel = le.fit_transform(maternal.RiskLevel)
        y = maternal.RiskLevel
        print(y.unique(), le.inverse_transform(y.unique()))
        X = maternal.loc[:, maternal.columns != 'RiskLevel']
        return self.data_preprocess(X, y)

    
    def tuning_gyperparameter(self, X_train, X_test, y_train, y_test, estimator_name): 
        self.estimator_name = estimator_name
        estimator = estimators[estimator_name]
        parameter_space = hyperparameters[estimator_name]
        if estimator_name != 'k-nearest neighbors':
            clf_func = estimator(random_state = random_seed)
        else:
            clf_func = estimator()
        # higher the AUC value for a classifier, the better its ability to distinguish
        clf = GridSearchCV(clf_func, parameter_space, n_jobs = n_jobs_val, cv = cv_splitter, scoring = score_method, return_train_score = True)
        clf.fit(X_train, y_train)
        # print(estimator_name + ' Best parameters found:\n', clf.best_params_)

        params = clf.cv_results_['params']
        train_means = clf.cv_results_['mean_train_score']
        test_means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        cv_results = pd.concat([pd.DataFrame(params), pd.DataFrame(list(zip(train_means, test_means, stds)), \
                                            columns =['train_score', 'test_score', 'std_test_score'])], axis = 1)
        cv_results['diff_score'] = abs(cv_results.train_score - cv_results.test_score)
        self_defined_best_params = clf.cv_results_['params'][cv_results.diff_score[cv_results.test_score>max(cv_results.test_score)*0.99].idxmin()]
        cv_results.to_csv(estimator_name + '.csv')
        
        plt.figure(figsize=(20, 6))
        plt.scatter(list(map(str, params)), train_means, label = 'train_score')
        plt.scatter(list(map(str, params)), test_means, label = 'test_score')
        plt.axvline(x = str(self_defined_best_params), ls = '--', label = 'best_param')
        plt.title(self.estimator_name + ' Hyperparameter tuning results', fontsize = 15)
        plt.ylabel('score', fontsize = 12)
        plt.xticks(rotation = 45, ha = 'right', fontsize = 12)
        plt.legend()
        plt.plot()
        
        self.estimator, self.best_params = estimator, self_defined_best_params#clf.best_params_
        return self.estimator, self.best_params #estimator(random_state = random_seed, **clf.best_params_)
    
        
    def plot_learning_curve(self, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 5),):
        if estimator_name != 'k-nearest neighbors':
            best_estimator = self.estimator(random_state = random_seed, **self.best_params)
        else:
            best_estimator = self.estimator(**self.best_params)
    
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            best_estimator, X_train, y_train, cv = cv_splitter, n_jobs = n_jobs_val,
            train_sizes = train_sizes, return_times = True, scoring = score_method)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
    
        # Plot learning curve
        _, ax = plt.subplots(1, 1)
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - 2 * train_scores_std,
            train_scores_mean + 2 * train_scores_std, alpha=0.1, color="r",)
        ax.fill_between(train_sizes, test_scores_mean - 2 * test_scores_std,
            test_scores_mean + 2 * test_scores_std, alpha=0.1, color="g", )
        ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        ax.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
        ax.legend(loc="best")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")    
        ax.set_title(self.estimator_name + " learning curve")
        
        plt.show()
        return
    
    
    @ignore_warnings(category=ConvergenceWarning)
    def plot_learning_curve_on_iteration(self, X_train, y_train):
        max_iter_num = best_params['max_iter']
        epochs = range(max_iter_num) if max_iter_num < 500 else range(0, max_iter_num + max_iter_num//10, max_iter_num//10)
        
        def training_cv_score(best_estimator, training_score_list, validation_score_list):
            y_train_pred = best_estimator.predict(x_training)
            # Multi-layer Perceptron classifier optimizes the log-loss function using LBFGS or stochastic gradient descent.
            curr_train_score = balanced_accuracy_score(y_training, y_train_pred) # training performances
            y_val_pred = best_estimator.predict(x_validation) 
            curr_valid_score = balanced_accuracy_score(y_validation, y_val_pred) # validation performances
            training_score_list.append(curr_train_score) # list of training perf to plot
            validation_score_list.append(curr_valid_score) # list of valid perf to plot
            
        training_score, validation_score = [], []        
        for train_index, test_index in cv_splitter.split(X_train):
            x_training, x_validation = X_train.iloc[train_index], X_train.iloc[test_index]
            y_training, y_validation = y_train.iloc[train_index], y_train.iloc[test_index]
            training_score_list, validation_score_list = [], []            
           
            if self.estimator_name == 'Neural networks':
                best_estimator = self.estimator(random_state = random_seed, **self.best_params)
                best_estimator.warm_start = True
                best_estimator.max_iter = 1 
                for epoch in epochs:       
                    best_estimator.partial_fit(x_training, y_training, classes = np.unique(y_training))                     
                    training_cv_score(best_estimator, training_score_list, validation_score_list)

            elif self.estimator_name == 'Support Vector Machines':
                for epoch in epochs:
                    best_estimator = self.estimator(random_state = random_seed, **self.best_params)
                    best_estimator.max_iter = epoch
                    best_estimator.fit(x_training, y_training)
                    training_cv_score(best_estimator, training_score_list, validation_score_list)
                    
            training_score.append(training_score_list)
            validation_score.append(validation_score_list)                    
                    
        plt.plot(epochs, np.array(training_score).mean(axis = 0), label = 'Training')
        plt.plot(epochs, np.array(validation_score).mean(axis = 0), label = 'Cross-validation')
        plt.title(self.estimator_name + ' Accuracy under cross validation')
        plt.xlabel('Iterations')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
    
        return training_score, validation_score
        
        
    def predict_test_data(self, X_train, X_test, y_train, y_test):
        if self.estimator_name == 'Neural networks':
            best_estimator = self.estimator(random_state = random_seed, **self.best_params)
        else:
            best_estimator = self.estimator(**self.best_params)
        best_estimator.fit(X_train, y_train)
        y_pred = best_estimator.predict(X_test)
        test_score = balanced_accuracy_score(y_test, y_pred)
        return test_score


if __name__ == "__main__":
    sl = supervised_learning(estimator_names)
    out_of_sample_scores = []
    for get_data in [sl.get_kepler_data, sl.get_maternal_data]:   
        X_train, X_test, y_train, y_test = get_data()     
        out_of_sample_score = []
        for estimator_name in sl.estimator_names:
            estimator, best_params = sl.tuning_gyperparameter(X_train, X_test, y_train, y_test, estimator_name)
            sl.best_hyperparameters[estimator_name] = best_params
            sl.plot_learning_curve(X_train, y_train)
            if estimator_name in iterative_algorithms:
                training_score, validation_score = sl.plot_learning_curve_on_iteration(X_train, y_train)
            test_score = sl.predict_test_data(X_train, X_test, y_train, y_test)
            out_of_sample_score.append(test_score)
        out_of_sample_scores.append(out_of_sample_score)
    
