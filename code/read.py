'该文件实现数据清洗、数据读取、验证集划分、验证集评估、提交文件生成'
import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime
np.random.seed(0)

def read(name) -> pd.DataFrame:
    try:
        filename = 'data/' + name + '.parquet'
        print(f'正在读取文件{filename}')
        df = pd.read_parquet(filename)
    except:
        print(f"读取文件{'data/' + name + '.parquet'}失败，尝试读取{'data/' + name + '.csv'}")
        filename = 'data/' + name + '.csv'
        print(f'正在读取文件{filename}')
        df = pd.read_csv(filename, index_col=0)
    return df

def get_data_sample(name, df = None):
    '生成一个有10000个样本的数据集'
    if df is None:
        df = read('data/' + name + '.parquet')
    sample = df.head(10000)
    save(sample, name + '_sample', csv = True, parquet = True) 
    return sample

def map_user_to_id(df: pd.Series, output_path = None, from_file = None):
    '''
    output_path为输出路径，保存用户名到id的映射，为none就不保存
    from_file为输入路径，如果为none就根据df计算
    '''
    print('正在将用户映射到id')
    if from_file is not None:
        mp = pd.read_csv('data/' + from_file + '_user_id_map.csv', index_col = 0)
    else:
        all_users = df.unique()
        size = len(all_users)
        mp = pd.DataFrame(np.arange(size), index = all_users, columns = ['ID'])
        mp.index.rename('user', inplace = True)
        
    if output_path:
        output_path = 'data/' + output_path + '_user_id_map.csv'
        mp.to_csv(output_path)

    if 0 not in df:
        '还没有映射过'
        df = df.map(mp['ID'])
    return df

# def map_id_to_user(df: pd.Series, name):
#     print('正在将id映射到用户')
#     mp = pd.read_csv('data/' + name + '_user_id_map.csv', index_col = 1)
#     df = df.map(mp['user'])
#     return df

def save(df: pd.DataFrame, name, csv = False, parquet = False):
    print(f'正在保存DataFrame: {name}')
    if csv:
        df.to_csv('data/' + name + '.csv')
    if parquet:
        df.to_parquet('data/' + name + '.parquet')

def map_user_to_id_and_save(name):
    '此函数映射后会直接覆盖'
    if name == 'train' or name == 'test':
        df = read(name)
        if not os.path.exists('data/' + name + '_user_id_map.csv'):
            df['customer_ID'] = map_user_to_id(df['customer_ID'], name)
            save(df, name, parquet = True)
            get_data_sample(name, df = df)
    elif name == 'train_labels' or name == 'test_labels':
        df = read(name)
        df.index = map_user_to_id(df.index, None, from_file=name.split('_')[0])
        save(df, name, csv = True)
    else:
        raise NameError('不正确的文件名')
        
def clean():
    # map_user_to_id_and_save('train')
    # map_user_to_id_and_save('train_labels')
    map_user_to_id_and_save('test')
    map_user_to_id_and_save('test_labels')
    
def read_train(sample = False):
    '读取训练集，返回(数据,标签)元组。注意训练集是一个时序序列，相同customer_ID是同一个样本，只对应一个标签'
    if not sample:
        x = read('train')
        y = read('train_labels')
    else:
        x = read('train_sample')
        y = read('train_labels').head(x.shape[0])
    x.set_index('customer_ID', inplace = True)
    return x, y

def read_test(sample = False, block = None):
    '读取测试集。注意训练集是一个时序序列，相同customer_ID是同一个样本，只需要输出一个标签'
    if not sample:
        if block is None:
            x = read('test')
            x.set_index('customer_ID', inplace=True)
        else:
            x = read(f'test_block{block}')
    else:
        x = read('test_sample')
        x.set_index('customer_ID', inplace=True)
    return x

if __name__ == '__main__':
    # clean()
    
    # get_submit_files(pd.read_csv('data/test_labels.csv'), 'test')
    
    pass