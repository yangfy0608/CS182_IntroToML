'数据预处理文件'
from read import *
from utils import *
from global_params import *
import pandas as pd
import numpy as np

def mean_preprocess(x: pd.DateOffset, y: pd.DataFrame = None) -> pd.DataFrame:
    print('正在预处理数据，时序序列处理：求平均')
    if y is not None:
        '删除日期行'
        del x['S_2']
        '将-1设为空值'
        x.replace(-1, np.nan, inplace = True)
        '删除缺失值超过20%的列'
        x.dropna(thresh = len(x) * 0.8, axis = 1, inplace = True)
        '删除所有值都一样的列'
        x.drop(x.columns[(x == x.iloc[0]).all()], axis = 1, inplace = True)
        '不同时间的特征取均值'
        x = x.groupby(x.index).agg('mean')
        '均值填充缺失值'
        x.fillna(x.mean(skipna = True), inplace = True)
        return x, y
    else:
        del x['S_2']
        x.replace(-1, np.nan, inplace = True)
        x = x.groupby(x.index).agg('mean')
        x.fillna(x.mean(skipna = True), inplace = True)
        x = x[features]
        return x
    
def statistics_preprocess(x: pd.DateOffset, y: pd.DataFrame = None) -> pd.DataFrame:
    print('正在预处理数据，时序序列处理：求不同统计值')
    if y is not None:
        '删除日期行'
        del x['S_2']
        '将-1设为空值'
        x.replace(-1, np.nan, inplace = True)
        '删除缺失值超过20%的列'
        x.dropna(thresh = len(x) * 0.8, axis = 1, inplace = True)
        '删除所有值都一样的列'
        x.drop(x.columns[(x - x.iloc[0] + 1e-8 < 1e-6).all()], axis = 1, inplace = True)
        '计算不同时间的特征的均值、方差、最大、最小、最后一个值'
        x = x.groupby(x.index).agg(['mean', 'std', 'min', 'max', 'last'])
        x.columns = ['_'.join(col) for col in x.columns]
        '删除所有值都一样的列'
        x.drop(x.columns[x.std() < 1e-2], axis = 1, inplace = True)
        '均值填充缺失值'
        x.fillna(x.mean(skipna = True), inplace = True)
        return x, y
    else:
        del x['S_2']
        x.replace(-1, np.nan, inplace = True)
        x = x.groupby(x.index).agg(['mean', 'std', 'min', 'max', 'last'])
        x.columns = ['_'.join(col) for col in x.columns]
        x.fillna(x.mean(skipna = True), inplace = True)
        x = x[features]
        return x

def get_preprocess_train_data(method = preprocess_method, to_numpy = True):
    print('正在读取预处理后的训练集')
    if method == 'statistics':
        preprocess_func = statistics_preprocess
    elif method == 'mean':
        preprocess_func = mean_preprocess
    else:
        raise ValueError('method必须为mean或statistics')
    
    if not sample:
        try:
            x = pd.read_parquet(f'processed_data/pre_{method}_train_x.parquet')
            y = pd.read_parquet(f'processed_data/pre_{method}_train_y.parquet')
        except:
            x, y = read_train(sample = False)
            x, y = preprocess_func(x, y)
            x.to_parquet(f'processed_data/pre_{method}_train_x.parquet')
            y.to_parquet(f'processed_data/pre_{method}_train_y.parquet')
        if feature_selection:
            _features = read_features(feature_selection_method)
            x = x[_features]
    else:
        x, y = read_train(sample)
        x, y = preprocess_func(x, y)
    global feature_size, features
    feature_size = x.shape[1]
    features = list(x.columns)
    if to_numpy:
        x = x.to_numpy()
        y = y.to_numpy()
    return x, y

def get_preprocess_test_data(method = preprocess_method, to_numpy = True):
    print('正在读取预处理后的测试集')
    if method == 'statistics':
        preprocess_func = statistics_preprocess
    elif method == 'mean':
        preprocess_func = mean_preprocess
    else:
        raise ValueError('method必须为mean或statistics')
    try:
        x = pd.read_parquet(f'processed_data/pre_{method}_test_x.parquet')
    except:
        x = read_test()
        x = preprocess_func(x)
        x.to_parquet(f'processed_data/pre_{method}_test_x.parquet')
    if not sample and feature_selection:
        _features = read_features(feature_selection_method)
        x = x[_features]
    if to_numpy:
        x = x.to_numpy()
    return x

