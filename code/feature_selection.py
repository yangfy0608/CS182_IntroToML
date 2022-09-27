'LASSO回归做递归特征消除'
from read import *
from utils import *
from preprocess import *
from global_params import *
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
import warnings
warnings.filterwarnings('ignore')

def train(train_x, train_y, method = 'lasso'):
    if method == 'lasso':
        model = Lasso(alpha=0.00002, fit_intercept=True, random_state=0, max_iter=10000)
    elif method == 'ridge':
        model = Ridge(alpha=0.0005, fit_intercept=True, random_state=0, max_iter=10000)
    elif method == 'logistic':
        model = LogisticRegression(penalty='l1', solver = 'saga', 
                                   C=1, random_state=0, max_iter=100, n_jobs=11, verbose=1)
    elif method == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("method must be 'lasso', 'ridge', 'logistic' or 'linear'")
    model.fit(train_x, train_y)
    # w = model.coef_
    # print(np.max(w), np.min(w), np.mean(w), np.std(w))
    return model

def predict(x, model, method = 'lasso'):
    if method == 'logistic':
        pred = model.predict_proba(x)
        pred = pred[:, 1 : 2]
    else: 
        pred = model.predict(x)
    pred = pred_ndarray_to_df(pred)
    # print(pred.head())
    return pred

def RFE(x: np.ndarray, y: np.ndarray, features, method = 'lasso'):
    '递归消除特征'
    print('开始递归消除特征')
    gc.collect()
    rest_features = features
    best_features = None
    best_score = 0.0
    score = 0.0
    K = 2 # K折检验
    N = 3 # 一次删除特征的数量
    while score > best_score - 0.0001:
        gc.collect()
        score = 0.0
        if len(rest_features) >= 100: N = 3
        else: N = 2
        kf = SKL_KFold(n_splits=K, shuffle=True, random_state=1)
        weights = np.zeros(len(rest_features), dtype=np.float32)
        for k, (train_index, test_index) in enumerate(kf.split(x)):
            train_x, train_y = x[train_index], y[train_index]
            test_x, test_y = x[test_index], y[test_index]
            mean = np.mean(train_x)
            std = np.std(train_x)
            train_x = (train_x - mean) / (std + 1e-9)
            test_x = (test_x - mean) / (std + 1e-9)
            model = train(train_x, train_y, method)
            if len(model.coef_.shape) == 2:
                weights += model.coef_.flatten() / K
            else:
                weights += model.coef_ / K
            pred = predict(test_x, model, method)
            print(f'剩余特征个数: {len(rest_features)} | flod: {k + 1} | ', end='')
            score += evaluate(pred, test_y)[0] / K
        print(f'K-Fold score : {score: .4f}')

        zeros = np.abs(weights) <= 1e-6
        N = max(N, np.sum(zeros))

        '按照特征重要性绝对值升序排序'        
        order = np.argsort(np.abs(weights))
        print(f'被删除特征权重: {weights[order[:N]]}')
        print(f'剩余特征个数: {len(rest_features) - N}')
        print(f'删除特征: {rest_features[order[:N]]}')
        x = np.delete(x, order[:N], axis=1)
        rest_features = rest_features.delete(order[:N])

        if score < best_score - 0.0001:
            '如果剩余特征的预测和最优的差距达到一定阈值，直接返回'
            break
        if score > best_score - 0.00002:
            best_score = score
            best_features = rest_features.copy()

    print(f'best features: {best_features}')
    return best_features

def Ridge_forward(x: np.ndarray, y: np.ndarray, features):
    pass

if __name__ == '__main__':
    method = 'lasso'
    savename = f'{method} + RFE'
    
    x, y = get_preprocess_train_data(to_numpy=False)
    features = x.columns
    x = x.to_numpy()
    y = y.to_numpy()
    
    features = RFE(x, y, features, method=method)
    size = len(features)
    save_features(features, f'{savename}({size})')
