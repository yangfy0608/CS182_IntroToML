from read import *
from utils import *
from global_params import *
from preprocess import *
import xgboost as xgb

xgb_params = { 
    'random_state' : 0,                 #* 随机种子
    'nthread' : 6,                      #* 线程数
    'verbosity' : 1,                    #* 日志显示程度 0-silent, 1-warning, 2-info, 3-debug

    'objective' : 'binary:logistic',    #* 问题类别
    'eval_metric' : 'logloss',          #* 评估指标 logloss | auc
    'tree_method' : 'gpu_hist',         #* 树构造方法
    'predictor' : 'gpu_predictor',      #* 预测器
    
    'max_depth' : 5,                    #* 树最大深度
    'subsample' : 0.8,                  #* 训练一个数时样本采样比例（减少过拟合）
    'colsample_bytree' : 0.6,           #* 训练一个数时特征采样比例（减少过拟合）
    'learning_rate' : 0.05,             #* 学习率
    'lambda' : 5,                       #* L2正则系数
    'scale_pos_weight' : 1,             #* 正样本权重
}

def train(train_x, train_y, valid_x = None, valid_y = None, save = None):
    print('开始训练')
    dtrain = xgb.DMatrix(train_x, label=train_y)
    evals = [(dtrain, 'train')]
    if valid_x is not None:
        dvalid = xgb.DMatrix(valid_x, label=valid_y)
        evals.append((dvalid, 'valid'))
        model = xgb.train(xgb_params, dtrain, num_boost_round=10000, 
                        evals=evals, early_stopping_rounds=100, verbose_eval=100)
    else:
        model = xgb.train(xgb_params, dtrain, evals=evals, 
                          num_boost_round=2000, verbose_eval=100)
    # feature_importance = model.get_score(importance_type='weight').copy()
    # feature_importance = pd.DataFrame(feature_importance, index=[0]).T
    # feature_importance.to_csv('feature_importance.csv')
    # assert False
    if save is not None:
        save_model(model, save)
    return model

def predict(x, model, to_df=True):
    '内存占用太大，这里分块预测'
    pred_all = []
    x = Loader(x)
    while not x.end():
        dtest = xgb.DMatrix(x.next())
        pred = model.predict(dtest)
        if len(pred) == 0: continue
        pred_all.append(pred)
    pred = np.concatenate(pred_all, axis = 0)
    if to_df:
        pred = pred_ndarray_to_df(pred)
    return pred

def test(x, model, method):
    print('开始测试')
    pred = predict(x, model)
    print(pred.head())
    get_submit_files(pred, method)

if __name__ == '__main__':
    method = method = get_model_name(train_method = 'xgboost')
    x, y = get_preprocess_train_data(to_numpy=False)
    print(f'feature size: {x.shape[1]}')
    
    if not submit:
        # train_x, train_y, valid_x, valid_y = divide_train_valid(x, y, 0.2)
        # model = train(train_x, train_y, valid_x, valid_y, save = f'{method}(part)')
        # pred = predict(valid_x, model)
        # evaluate(pred, valid_y)
        result = KFold(x, y, train, predict, seed=0, save=method, normalize=False)
        save_param_and_result(method, xgb_params, result)
        
    else:
        result = KFold(x, y, train, predict, seed=0, save=method, normalize=False)
        save_param_and_result(method, xgb_params, result)
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
