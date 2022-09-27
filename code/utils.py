'提供一些辅助函数'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as SKL_KFold
import matplotlib.pyplot as plt
from read import *
from global_params import *
import gc
import pickle
import time

def divide_train_valid(x: np.ndarray, y: np.ndarray, valid_rate: float):
    '划分训练集和验证集'
    print('正在划分训练集和验证集')
    train_x, valid_x, train_y, valid_y = \
        train_test_split(x, y, test_size = valid_rate, shuffle = True, random_state = 0)
    return train_x, train_y, valid_x, valid_y

def evaluate(y_pred: pd.DataFrame, y_true: pd.DataFrame):
    '''
    评估标准 
    G = 2 * AUC - 1 (基尼系数)
    D = 预测为1的概率小于0.04的正例数 / 总正例数 (4%处召回率)
    M = 0.5 * (G + D)
    实际比赛的测试集的负例被采样为5%，权重为20倍。
    返回 score, G(基尼系数), D(4%处召回率)
    ! *此部分代码由比赛举办方给出*
    '''
    if isinstance(y_pred, np.ndarray):
        y_pred = pred_ndarray_to_df(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = pred_ndarray_to_df(y_true, 'target')
    y_true.index = y_pred.index
    
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    gini = normalized_weighted_gini(y_true, y_pred)
    recall = top_four_percent_captured(y_true, y_pred)
    score = 0.5 * (gini + recall)

    print(f'score = {score:.4f} | gini = {gini:.5f} | 4% recall = {recall:.5f}')
    return score, gini, recall

def pred_ndarray_to_df(y: np.ndarray, col_name = 'prediction') -> pd.DataFrame:
    '将预测结果转换为DataFrame'
    res = pd.DataFrame(y, columns=[col_name])
    res.index.rename('customer_ID', inplace=True)
    return res

def KFold(x, y, train_func, pred_func, eval_func = evaluate, 
          normalize = True, seed = 0, save: str = None):
    '''
    K折检验
    train_func输入(train_x,train_y,test_x,test_y)，输出model
    pred_func输入(test_x,model)，输出pred
    eval_func输入(pred,test_y)，输出score，默认为比赛官方给出的评估函数
    '''
    kf = SKL_KFold(n_splits=K, shuffle=True, random_state=seed)
    avg_score = 0.0
    avg_gini = 0.0
    avg_recall = 0.0
    for k, (train_index, test_index) in enumerate(kf.split(x)):
        gc.collect()
        if isinstance(x, pd.DataFrame):
            train_x, train_y = x.iloc[train_index], y.iloc[train_index]
            test_x, test_y = x.iloc[test_index], y.iloc[test_index]
        else:
            train_x, train_y = x[train_index], y[train_index]
            test_x, test_y = x[test_index], y[test_index]
        if normalize:
            mean = train_x.mean()
            std = train_x.std()
            train_x = (train_x - mean) / std
            test_x = (test_x - mean) / std
        try:
            model = train_func(train_x, train_y, test_x, test_y)
        except:
            model = train_func(train_x, train_y)
        if save is not None:
            save_model(model, method = save, fold = k)
        y_pred = pred_func(test_x, model)
        if isinstance(test_y, np.ndarray):
            test_y = pred_ndarray_to_df(test_y, col_name='target')
        test_y.index = y_pred.index
        print(f'fold {k + 1: d} | ', end = "")
        score, g, d = eval_func(y_pred, test_y)
        avg_score += score
        avg_gini += g
        avg_recall += d
        
    avg_score /= K
    avg_gini /= K
    avg_recall /= K
    print(f'score = {avg_score:.4f} | gini = {avg_gini:.4f} | 4% recall = {avg_recall:.4f}')
    return avg_score, avg_gini, avg_recall

def save_model(model, method, fold=None):
    if fold is None:
        pickle.dump(model, open(f'model/{method}.pkl', 'wb'))
    else:
        pickle.dump(model, open(f'model/{method}_fold{fold}.pkl', 'wb'))

def load_model(method, fold=None):
    if fold is None:
        model = pickle.load(open(f'model/{method}.pkl', 'rb'))
    else:
        model = pickle.load(open(f'model/{method}_fold{fold}.pkl', 'rb'))
    return model

def show_pred(pred: pd.DataFrame, verbose = True, save: str = None):
    '展示预测结果直方图'
    plt.hist(pred, bins=100)
    plt.title('Predictions')
    if verbose:
        plt.show()
    if save is not None:
        plt.savefig(f'image/{save}.png')

class Loader():
    '内存不够，分块加载数据'
    def __init__(self, x, blocks = 8):
        self.x = x
        self.blocks = blocks
        self.n = len(x)
        self.block_size = int(self.n / self.blocks) + 1
        self.iter = 0
    def next(self):
        l = self.iter * self.block_size
        r = min(self.n, (self.iter + 1) * self.block_size)
        self.iter += 1
        if isinstance(self.x, pd.DataFrame):
            return self.x.iloc[l:r]
        else:
            return self.x[l:r]
    def end(self):
        return self.iter == self.blocks

def get_model_name(train_method, show = True) -> str:
    '返回模型名称'
    _preprocess = f'预处理({preprocess_method})'
    if feature_selection and not sample:
        '如果是测试样本，不做特征选择'
        _feature_selection = f'特征选择({feature_selection_method})'
    _train_model = f'{train_method}'
    if sample:
        _train_model += '(sample)'
    try:
        res = f'{_preprocess} + {_feature_selection} + {_train_model}'
    except:
        res = f'{_preprocess} + {_train_model}'
    if show:
        print(f'模型名称：{res}')
    return res
    
def save_features(features, method):
    filename = f'features/{method}.txt'
    with open(filename, 'w') as f:
        f.write(str(list(features)))

def read_features(method):
    filename = f'features/{method}.txt'
    with open(filename, 'r') as f:
        features = eval(f.read())
    return features

def get_submit_files(pred: pd.DataFrame, method: str):
    '生成提交文件'
    df = pd.read_csv('prediction/sample_submit.csv', index_col = 0)
    df['prediction'] = pred['prediction'].values
    path = f'prediction/{method} # {datetime.now().strftime("%m-%d %H-%M")}.csv'
    df.to_csv(path)
    
def save_param_and_result(method: str, params: dict, result: tuple, explainaion: str = None):
    "保存参数结果, result为('score', 'gini', '4% recall')三元组"
    if sample: return # 小样本仅用于测试代码能否正确运行
    with open(f'调参记录/{method}.txt', 'a') as f:
        f.write(datetime.now().strftime(r"%m/%d %H:%M:%S"))
        f.write(f'\nmethod: {method}\n\n')
        f.write('params: \n')
        assert len(result) == 3
        result = dict(zip(('score', 'gini', '4% recall'), result))
        for key, value in params.items():
            f.write(f'\t{key} : {value}\n')
        f.write('\nresult: \n')
        for key, value in result.items():
            f.write(f'\t{key} : {value}\n')
        if explainaion is not None:
            f.write(f'\nexplaination: \n')
            f.write(explainaion + '\n')
        f.write('\n' + '-' * 50 + '\n')
        

if __name__ == '__main__':
    # x, y = read_train()
    # print(x.head(), y.head(), sep='\n')
    # train_x, train_y, valid_x, valid_y = divide_train_valid(x, y, 0.8)
    # print(train_x.head())
    # print(train_y.head())
    # print(valid_x.head())
    # print(valid_y.head())
    
    save_param_and_result('test', {'a': 1, 'b': 2}, (123, 123, 23))