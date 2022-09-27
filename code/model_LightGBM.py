from read import *
from utils import *
from global_params import *
from preprocess import *
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

lgb_params = {
    'random_state' : 0,             #* 随机种子
    'nthread' : 6,                  #* 线程数
    'verbosity' : -1,               #* 日志显示程度
    'device' : 'gpu',               #* 设备
    
    'objective': 'binary',          #* 问题类别
    'metric': 'binary_logloss',     #* 评估函数 binary_logloss | auc
    'boosting': 'dart',             #* 基分类器 gbdt | dart | goss
                                    #! goss算法是对梯度的单边采样，效率高，但准确度略低
                                    #! dart算法就是带dropout的gbdt，效率低，但准确度最高
    'drop_rate' : 0.1,              #! dropout比例，仅适用于dart基学习器，为0是就是gbdt
    'early_stopping_round' : 100,   #* 运行验证集变差次数
    'num_iteration' : 4000,         #* 迭代次数
    
    'max_bin' : 255,                #* 最大特征分块数
    'max_depth' : 5,                #* 树最大深度
    'min_child_samples' : 100,      #* 每个叶节点最少样本数
    'subsample' : 0.8,              #* 训练一个数时样本采样比例（减少过拟合）
    'colsample_bytree' : 0.6,       #* 训练一个数时特征采样比例（减少过拟合）
    'learning_rate' : 0.05,         #* 学习率
    'lambda_l2' : 5,                #* L2正则系数
}

def train(train_x, train_y, valid_x = None, valid_y = None, save = None):
    print('开始训练')
    dtrain = lgb.Dataset(train_x, label=train_y)
    data = [dtrain]
    data_name = ['train']
    if valid_x is not None:
        dvalid = lgb.Dataset(valid_x, label=valid_y)
        data.append(dvalid)
        data_name.append('valid')
        model = lgb.train(lgb_params, dtrain, valid_sets=data, 
                          valid_names=data_name, verbose_eval=100)
    else:
        model = lgb.train(lgb_params, dtrain, valid_sets=data, 
                          valid_names=data_name, verbose_eval=100)

    if save is not None:
        save_model(model, save)
    return model

def predict(x, model, to_df=True):
    '内存占用太大，这里分块预测'
    pred_all = []
    x = Loader(x)
    while not x.end():
        dtest = x.next()
        if len(dtest) == 0: continue
        pred = model.predict(dtest)
        pred_all.append(pred)
    pred = np.concatenate(pred_all, axis = 0)
    if to_df:
        pred = pred_ndarray_to_df(pred)
    return pred

if __name__ == '__main__':
    boost = lgb_params['boosting']
    method = get_model_name(train_method = f'LightGBM({boost})')
    x, y = get_preprocess_train_data(to_numpy=False)
    print(f'feature size: {x.shape[1]}')
    
    if not submit:
        # train_x, train_y, valid_x, valid_y = divide_train_valid(x, y, 0.2)
        # model = train(train_x, train_y, valid_x, valid_y, save = f'{method}(part)')
        # pred = predict(valid_x, model)
        # evaluate(pred, valid_y)
        result = KFold(x, y, train, predict, seed=0, normalize=False, save=method)
        save_param_and_result(method, lgb_params, result)
        
    else:
        result = KFold(x, y, train, predict, seed=0, save=method, normalize=False)
        save_param_and_result(method, lgb_params, result)
        del x, y
        gc.collect()
        test_x = get_preprocess_test_data(to_numpy = False)
        pred = []
        for k in range(K):
            print(f'正在使用flod-{k}模型预测')
            model = load_model(method=method, fold=k)
            pred.append(predict(test_x, model, to_df=False))
        pred = np.mean(pred, axis=0)
        pred = pred_ndarray_to_df(pred)
        show_pred(pred, verbose=False, save=method)
        get_submit_files(pred, method)
