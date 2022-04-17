import pandas as pd
import numpy as np
from pycaret.classification import *

import matplotlib.pyplot as plt
import seaborn as sns

## EDA
# 最低限のデータ形式・基礎統計量の確認
def check_basic_data_points(df):
    print('# Shape')
    print(df.shape)
    print('------------------------------')
    print('# Num of null')
    print(df.isnull().sum())
    print('------------------------------')
    print('# Numeric summary')
    display(df.describe())
    print('------------------------------')
    print('# Categorical summary')
    display(df.describe(include='O'))
    return

# 数値型のカラムの可視化
def eda_numerical_feature(df, col, target_col, problem_type='clf'):
    all_df = df.copy()
    train_df = all_df[all_df['flg']=='train']
    test_df = all_df[all_df['flg']=='test']
    
    print('# About [{}]'.format(col))
    
    print('# Distribution')
    plt.figure(figsize=(6, 4))
    sns.distplot(train_df[col], norm_hist=False, kde=False, color='lightblue', label='train')
    sns.distplot(test_df[col], norm_hist=False, kde=False, color='orange', label='test')
    plt.title(col)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print()
    
    print('# Correlation')
    plt.figure(figsize=(6, 4))
    if problem_type == 'clf':
        sns.boxplot(x=target_col, y=col, data=train_df)
    elif problem_type == 'reg':
        sns.scatterplot(x=target_col, y=col, data=train_df)
    plt.title(col)
    plt.tight_layout()
    plt.show()
    print()
    
    print('------------------------------------------------------------')

    return

# カテゴリ型のカラムの可視化
def eda_categorical_feature(df, col, id_col, target_col, problem_type='clf'):
    all_df = df.copy()
    train_df = all_df[all_df['flg']=='train']
    test_df = all_df[all_df['flg']=='test']
    
    train_classes = set(train_df[col])
    test_classes = set(test_df[col])
    all_classes = train_classes | test_classes
    
    print('# About [{}]'.format(col))
    
    # クラス数が多すぎる場合、スキップ
    if len(all_classes) > 30:
        print('Skip because of too many classes ({}).'.format(len(all_classes)))
        print('------------------------------------------------------------')
        return

    print('# Category Distribution')
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=all_df, hue='flg')
    plt.title(col)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    print('------------------------------------------------------------')

    print('Correlation')
    plt.figure(figsize=(6, 4))
    if problem_type == 'clf':
        count_df = train_df.pivot_table(index=col, 
                                        columns=target_col, 
                                        values=id_col, 
                                        aggfunc='count').fillna(0).astype(int)
        # display(count_df)
        sns.heatmap(count_df, cmap='Greens', annot=True, fmt='d')
    elif problem_type == 'reg':
        sns.boxplot(x=target_col, y=col, data=train_df)
    plt.title(col)
    plt.tight_layout()
    plt.show()
    print()

    print('------------------------------------------------------------')
   
    return