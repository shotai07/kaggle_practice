from typing import Dict
import pandas as pd
import numpy as np

def _len_str(x) -> int:
    return len(str(x))

def _left_str(x, num_chars: int) -> int:
    calib_num_chars = num_chars - 1
    return str(x)[:calib_num_chars]

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    Y_COL = parameters['Y_COL']
    
    train_df = train_df.copy().assign(flg='train')
    test_df = test_df.copy().assign(flg='test')
    test_df[Y_COL] = None

    prepro_df = pd.concat([train_df, test_df])

    # Name
    # 欠損値補間
    prepro_df['fix_Name'] = prepro_df['Name'].fillna('Null')
    #* 文字数
    prepro_df['len_Name'] = prepro_df['fix_Name'].map(_len_str)

    # Ticket
    # 欠損値補間
    prepro_df['fix_Ticket'] = prepro_df['Ticket'].fillna('-1')
    # 文字数
    prepro_df['len_Ticket'] = prepro_df['fix_Ticket'].map(_len_str)
    # 最初の2文字
    prepro_df['first_chars_Ticket'] = prepro_df['fix_Ticket'].map(lambda x: _left_str(x, 2))

    # Cabin
    # 欠損値補間
    prepro_df['fix_Cabin'] = prepro_df['Cabin'].fillna('-1')
    # 文字数
    prepro_df['len_Cabin'] = prepro_df['fix_Cabin'].map(_len_str)
    # 最初の1文字
    prepro_df['first_chars_Cabin'] = prepro_df['fix_Cabin'].map(lambda x: _left_str(x, 1))

    return prepro_df