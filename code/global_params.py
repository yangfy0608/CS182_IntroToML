'该文件有一些全局参数'

'sample为完整数据集10000行'
'submit选项表明是否生成提交到kaggle的预测结果'
'注意：sample为True时，submit默认为False'
sample = True
submit = False

'预处理方法 目前仅有 mean | statistics'
preprocess_method = 'statistics'

'特征选择方法，注意：feature_selection为False时，method默认为None'
feature_selection = True
feature_selection_method = 'lasso + RFE(441)'
    
'K折检验'
K = 5
    
if sample:
    submit = False
    feature_selection = False
    
if feature_selection == False:
    select_features_method = None