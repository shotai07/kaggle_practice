import numpy as np
import pandas as pd
import re

import os

import warnings
warnings.filterwarnings('ignore')

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# model
import lightgbm as lgb
import shap

class DS_PREPROCESS():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._descriminate_features()
        return

    def _descriminate_features(self):
        self.categorical_feature_names = []
        self.numerical_feature_names = []

        for c, t in zip(self.x.columns, self.x.dtypes):
            if t == 'object':
                self.categorical_feature_names.append(c)
            else:
                self.numerical_feature_names.append(c)
        return

    def show_desc_features(self):
        print('Categorical')
        print(self.categorical_feature_names)
        print('Numerical')
        print(self.numerical_feature_names)
        return

    def get_desc_features(self):
        return {'cat': self.categorical_feature_names, 'num': self.numerical_feature_names}

    def label_encode_x(self, x=None):
        if x is None:
            x = self.x

        for c, t in zip(x.columns, x.dtypes):
            if t == 'object':
                x[c] = x[c].astype(str)
                labels = x[c]
                if labels.isna().sum() == 0:
                    labels.append(None)

                le = LabelEncoder()
                le.fit(labels)
                x[c] = le.transform(x[c])
        return x

    def check_null_x(self, x=None):
        if x is None:
            x = self.x
        return x.isnull().sum()

class DS_MODEL():
    def __init__(self):
        self.base_classify_estimators = {
            'lgb': lgb.LGBMClassifier(num_leaves=100, learning_rate=0.1, min_split_gain=0),
            'rbf_svm': SVC(kernel='rbf', random_state=1),
            'linear_svm': SVC(kernel='linear', random_state=1),
            'logit': LogisticRegression(random_state=1),
            'dt': DecisionTreeClassifier(random_state=1),
            'rf': RandomForestClassifier(n_estimators=100, random_state=1),
            'nb': GaussianNB(),
            'knn': KNeighborsClassifier(),
        }
        self.classify_estimators = {}

        self.base_regression_estimators = {
            'lgb': lgb.LGBMRegressor(num_leaves=100, learning_rate=.1, min_split_gain=0)
        }
        self.regression_estimators = {}
        return

    def class_fit_predict(self, x_train, x_test, y_train, y_test, est_name, report_flg=True):
        if est_name == 'vote':
            if len(self.classify_estimators.keys()) > 2:
                model = VotingClassifier(estimators=self.classify_estimators.items())
            else:
                print('Caution: No models')
                return
        else:
            model = self.base_classify_estimators[est_name]

        model.fit(x_train, y_train)

        # predict test data
        y_pred = model.predict(x_test)

        # report scores
        if report_flg == True:
            self.class_score_report(y_test, y_pred)

        # add model to dict
        self.classify_estimators[est_name] = model
        return y_pred

    def class_fit_predict_cv(self, est_name, x=None, y=None, report_flg=True):
        if est_name == 'vote':
            if len(self.classify_estimators.keys()) > 2:
                model = VotingClassifier(estimators=self.classify_estimators.items())
            else:
                print('Caution: No models')
                return
        else:
            model = self.base_classify_estimators[est_name]

        scores_list = {
            'auc': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        for train_idx, test_idx in StratifiedKFold(n_splits=5).split(x, y):
            x_train = x.loc[train_idx, :]
            y_train = y[train_idx]
            x_test = x.loc[test_idx, :]
            y_test = y[test_idx]

            model.fit(x_train, y_train)

            # predict test data
            y_pred = model.predict(x_test)
            scores = self.calc_class_scores(y_test, y_pred)

            for k in scores_list.keys():
                scores_list[k].append(scores[k])

        for k in scores_list.keys():
            print(k + ': %.2f' % np.mean(scores_list[k]))

        # # add model to dict
        # self.classifier_estimators[est_name] = model
        return y_pred

    def calc_class_scores(self, y_test, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr, tpr)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        return {'auc': auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def class_score_report(self, y_test, y_pred):
        # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        # print("\nClassification Report:\n", classification_report(y_test, y_pred))
        scores = calc_class_scores(y_test, y_pred)
        print('AUC: %.2f' % scores['auc'])
        print('Accuracy: %.2f' % scores['accuracy'])
        print('Precision: %.2f' % scores['precision'])
        print('Recall: %.2f' % scores['recall'])
        print('F1: %.2f' % scores['f1'])
        return

    def reg_fit_predict(self, x_train, x_test, y_train, y_test, est_name, report_flg=True):
        if est_name == 'vote':
            if len(self.classify_estimators.keys()) > 2:
                model = VotingClassifier(estimators=self.classify_estimators.items())
            else:
                print('Caution: No models')
                return
        else:
            model = self.base_regression_estimators[est_name]

        model.fit(x_train, y_train)

        # predict test data
        y_pred = model.predict(x_test)

        # report scores
        if report_flg == True:
            self.reg_score_report(y_test, y_pred)

        # add model to dict
        self.regression_estimators[est_name] = model
        return

    def calc_reg_scores(self, y_test, y_pred):
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mae_mean = mae / y_test.mean()
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rmse_mean = rmse / y_test.mean()
        return {'mae': mae, 'mae_mean': mae_mean, 'rmse': rmse, 'rmse_mean': rmse_mean}

    def reg_score_report(self, y_test, y_pred):
        scores = self.calc_reg_scores(y_test, y_pred)
        print('MAE: %.2f' % scores['mae'])
        print('MAE/MEAN: %.4f' % scores['mae_mean'])
        print('RMSE: %.2f' % scores['rmse'])
        print('RMSE/MEAN: %.4f' % scores['rmse_mean'])
        return

    def get_estimators(self, type='classification'):
        if type=='classification':
            return self.classify_estimators
        else:
            return

class DS_SHAP():
    def __init__(self, model, x):
        self.model = model
        self.x = x
        self.feature_cols = self.x.columns
        shap.initjs()
        self.explainer = shap.TreeExplainer(model=self.model, feature_perturbation='tree_path_dependent', model_output='margin')
        self.shap_values = self.explainer.shap_values(X=self.x)
        return

    def show_summary_plot(self, plot_type='bar'):
        shap.summary_plot(self.shap_values, self.x, plot_type=plot_type)
        return

    def show_all_dependence_plot(self, type='classification'):
        for f in self.feature_cols:
            if type == 'classsification':
                shap.dependence_plot(ind=f, shap_values=self.shap_values[1], features=self.x, interaction_index=f)
            elif type == 'regression':
                shap.dependence_plot(ind=f, shap_values=self.shap_values, features=self.x, interaction_index=f)
        return