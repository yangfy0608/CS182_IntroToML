'模型堆叠'
from read import *
from utils import *
from preprocess import *
from global_params import *
import model_ANN
from model_ANN import ANN, ANN_norm, ANN_norm_res_dropout, Logistic
import model_xgboost
import model_LightGBM
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

ens_params = {
    'models' : ['xgboost', 'LightGBM(dart)', 'ANN_norm_res_dropout',], 
    # 'xgboost', 'LightGBM(dart)', 'LightGBM(gbdt)'
    # 'Logistic', 'ANN', 'ANN_norm', 'ANN_norm_res_dropout'
    'ens_method' : 'mlp', 
}

def get_models(methods : list, fold : int) -> dict:
    '读取对应的模型，返回一个字典'
    models = {}
    print(f'正在读取不同的模型{methods}')
    for method in methods:
        model_name = get_model_name(train_method = method, show = False)
        model = load_model(model_name, fold)
        models[method] = model
    return models

def get_predictions(models : dict, x):
    print('正在使用不同预测模型(元学习器)预测')
    pred_all = []
    for key, model in models.items():
        if 'ANN' in key or 'Logistic' in key:
            pred = model_ANN.predict(x, model)
        elif 'xgboost' in key:
            pred = model_xgboost.predict(x, model)
        elif 'LightGBM' in key:
            pred = model_LightGBM.predict(x, model)
        else:
            raise ValueError(f'未知的模型: {key}')
        pred: pd.DataFrame
        pred.rename(columns={'prediction' : key}, inplace=True)
        pred_all.append(pred)
    '按列合并'
    pred = pd.concat(pred_all, axis=1)
    # print('不同预测模型的相关系数：')
    # print(pred.corr())
    return pred

def ens_train(y_pred, y_true, ens_method):
    print('正在训练二级学习器')
    if ens_method == 'linear':
        '用线性回归计算不同模型权重，注意所有模型必须有非负权重'
        model = LinearRegression(fit_intercept=True, positive=True)
    elif ens_method == 'ridge':
        model = Ridge(alpha=0.02, positive=True, 
                      fit_intercept=True, random_state=0)
    elif ens_method == 'svr':
        model = SVR(C=1, epsilon=0.1, kernel='rbf', 
                    gamma='auto', degree=3)
    elif ens_method == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(10, 10), random_state=0)
    else:
        raise ValueError(f'不支持的模型堆叠方法：{ens_method}')
    model.fit(y_pred, y_true)
    print(y_pred.head())
    pred = model.predict(y_pred)
    pred = pred_ndarray_to_df(pred)
    print(pred.head())
    # print(f'二级学习器({ens_method})权重：\n{model.coef_}') # linear / ridge
    return model

def predict(meta_models, ens_model, x):
    preds = get_predictions(meta_models, x)
    pred = ens_model.predict(preds)
    pred_ndarray_to_df(pred)
    return pred

def ensebmle(methods : list, x, y, ens_method, total_method):
    '''
    模型堆叠
    methods为元学习器方法
    ens_method为二级学习器方法
    total_method为前两者合并
    '''
    kf = SKL_KFold(n_splits=K, shuffle=True, random_state=0)
    avg_score = 0.0
    avg_gini = 0.0
    avg_recall = 0.0
    for k, (_, test_index) in enumerate(kf.split(x)):
        '训练集用于训练元学习器了，在模型堆叠里不再使用，以免过拟合'
        gc.collect()
        ens_x, ens_y = x.iloc[test_index], y.iloc[test_index]
        models = get_models(methods, k)
        '用训练元学习器的验证集来集成模型'
        '如果submit=True，则使用全部验证集样本来集成模型'
        '如果submit=False，则使用70%验证集作为模型堆叠的训练集，其余的作为验证集'
        if submit:
            '还未完成'
            pass
        ens_train_x, ens_train_y, ens_valid_x, ens_valid_y = \
            divide_train_valid(ens_x, ens_y, valid_rate=0.3)
        train_pred = get_predictions(models, ens_train_x)
        ens_model = ens_train(train_pred, ens_train_y, ens_method)
        save_model(ens_model, method = total_method, fold = k)
        ens_pred = predict(models, ens_model, ens_valid_x)
        print(f'fold {k + 1: d} | ', end = "")
        score, g, d = evaluate(ens_pred, ens_valid_y)
        avg_score += score
        avg_gini += g
        avg_recall += d
        
    avg_score /= K
    avg_gini /= K
    avg_recall /= K
    print(f'score = {avg_score:.4f} | gini = {avg_gini:.4f} | 4% recall = {avg_recall:.4f}')
    return avg_score, avg_gini, avg_recall
    
if __name__ == '__main__':
    methods = ens_params['models']
    ens_method = ens_params['ens_method']
    
    total_method = get_model_name(f'ensebmle')
    x, y = get_preprocess_train_data(to_numpy=False)
    result = ensebmle(methods, x, y, ens_method, total_method)
    save_param_and_result(total_method, ens_params, result)
    del x, y
    gc.collect()
    
    if submit:
        test_x = get_preprocess_test_data(to_numpy = False)
        pred = []
        for k in range(K):
            print(f'正在使用flod-{k}模型预测')
            models = get_models(methods, fold = k)
            ens_model = load_model(ens_method, fold = k)
            pred.append(predict(models, ens_model, test_x))
        pred = np.mean(pred, axis=0)
        pred = pred_ndarray_to_df(pred)
        show_pred(pred, verbose=False, save=total_method)
        get_submit_files(pred, ens_method)
        