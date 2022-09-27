from read import *
from utils import *
from preprocess import *
from global_params import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
np.random.seed(0)

ANN_params = {
    #* 'Logistic', 'ANN', 'ANN_norm', 'ANN_norm_res_dropout'
    'model' : 'ANN_norm_res_dropout', 
    'batch_size' : 4196,
}
batch_size = ANN_params['batch_size']

class Amex_Dataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        self.x = torch.FloatTensor(self.x)
        if y is not None:
            self.y = torch.FloatTensor(self.y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        x = self.x[i]
        if self.y is not None:
            y = self.y[i]
            return x, y
        else:
            return x

class Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.logistic = nn.Sequential(
            nn.Linear(feature_size, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.logistic(x)

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(feature_size, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.nn(x)

class ANN_norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
        nn.Linear(feature_size, 100),
        nn.LayerNorm(100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.LayerNorm(50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.LayerNorm(20),
        nn.Linear(20, 1),
        nn.Sigmoid()
    )
    def forward(self, x):
        return self.nn(x)

class ANN_norm_res_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn1 = self.Linear(feature_size, 128)
        self.nn2 = nn.Sequential(
            self.Linear(128, 32),
            self.Linear(32, 32)
        )
        self.nn3 = nn.Sequential(
            nn.Dropout(0.5), 
            self.Linear(160, 16),
            self.Linear(16, 1, norm = False, activation = nn.Sigmoid())
        )
    def Linear(self, input_dim: int, output_dim: int, norm = True, 
               dropout = False, activation = nn.ReLU()):
        layers = [nn.Linear(input_dim, output_dim)]
        if norm: layers.append(nn.LayerNorm(output_dim))
        if dropout: layers.append(nn.Dropout(0.5))
        layers.append(activation)
        return nn.Sequential(*layers)
    def forward(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x1)
        x = torch.cat((x1, x2), dim = 1)
        x = self.nn3(x)
        return x

def train(x: np.ndarray, y: np.ndarray, model: nn.modules, 
          valid_x = None, valid_y = None, 
          loss_func = nn.BCELoss, lr = 0.1, epoch = 20, save = None) -> nn.Module:
    print('开始训练')
    train_set = Amex_Dataset(x, y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = model().cuda()
    loss_func = loss_func()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for k in range(epoch):
        start_time = time.time()
        loss = 0.0

        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x.cuda())
            batch_loss = loss_func(pred, y.cuda())
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        
        print('epoch: %2d | time: %.2fs | train_loss: %.6f' % \
            (k + 1, time.time() - start_time, loss / len(train_loader)))
    if save is not None:
        save_model(model, save)
    return model

def predict(x, model, to_df=True) -> np.ndarray:
    model.eval()
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    pred = []
    data_set = Amex_Dataset(x)
    del x
    gc.collect()
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            y = model(x)
            pred += y
    pred = torch.cat(pred, dim = 0)
    pred = pred.cpu().data.numpy()
    if to_df:
        pred = pred_ndarray_to_df(pred)
    return pred

# def test(x: np.ndarray, model, method):
#     print('开始测试')
#     pred = predict(x, model)
#     print(pred.head())
#     get_submit_files(pred, method)

if __name__ == '__main__':
    if ANN_params['model'] == 'Logistic':
        model = Logistic
        epoch = 5
    elif ANN_params['model'] == 'ANN':
        model = ANN
        epoch = 10
    elif ANN_params['model'] == 'ANN_norm':
        model = ANN_norm
        epoch = 20
    elif ANN_params['model'] == 'ANN_norm_res_dropout':
        model = ANN_norm_res_dropout
        epoch = 20
    else:
        raise ValueError('model not found')
    ANN_params['epoch'] = epoch
    import functools
    train = functools.partial(train, model = model, epoch = epoch)
    
    method = get_model_name(train_method = ANN_params['model'])
    
    x, y = get_preprocess_train_data()
    feature_size = x.shape[1]
    print(f'feature size: {x.shape[1]}')
    
    if not submit:
        # train_x, train_y, valid_x, valid_y = divide_train_valid(x, y, 0.2)
        # mean = np.mean(train_x, axis = 0)
        # std = np.mean(train_x, axis = 0)

        # train_x = (train_x - mean) / (std + 1e-6)
        # valid_x = (valid_x - mean) / (std + 1e-6)

        # train(train_x, train_y, model, save=method + '(part)', epoch = epoch)
        result = KFold(x, y, train, predict, seed=0, normalize=True, save=method)
        save_param_and_result(method, ANN_params, result)

    if submit:
        '使用测试集，生成准备提交到kaggle的预测结果'
        # train_x, train_y = x, y
        # mean = np.mean(train_x, axis = 0)
        # std = np.mean(train_x, axis = 0)
        # train_x = (train_x - mean) / (std + 1e-6)
        # model = train(train_x, train_y, model, save=method, epoch = epoch)
        # del train_x, train_y, x, y # 节约内存
        # gc.collect()
        
        # test_x = get_preprocess_test_data()
        # test_x = (test_x - mean) / (std + 1e-6)
        # test(test_x, model, method)
        
        result = KFold(x, y, train, predict, seed=0, save=method, normalize=True)
        save_param_and_result(method, ANN_params, result)
        del x, y
        gc.collect()
        test_x = get_preprocess_test_data()
        pred = []
        for k in range(K):
            print(f'正在使用flod-{k}模型预测')
            model = load_model(method=method, fold=k)
            pred.append(predict(test_x, model, to_df=False))
        pred = np.mean(pred, axis=0)
        pred = pred_ndarray_to_df(pred)
        show_pred(pred, verbose=False, save=method)
        get_submit_files(pred, method)
