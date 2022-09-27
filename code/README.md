# Introduction to Machine Learning (CS182) Course Project
This is the code portion of the course project for Introduction to Machine Learning (CS182). Our task is credit card default prediction, and the dataset is from a kaggle competition https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514. The code includes ANN, LightGBM, xgboost, feature selection, model stacking part.

Note: The original dataset was too large, our submission only included a small sample of 10,000 rows. Please make sure sample=True in global_params.py when running the code if your do not download the total dataset.

## File Structure
```
|- data/           * data sample
|- features/       * result of feature selection
|- image/          * images. For data sample, it should be empty.
|- model/          * models
|- processed_data/ * Process file. For data sample, it should be empty.
|- prediction/     * submit to kaggle. For data sample, it should be empty.
|- 调参记录/        * After running the code, the result will be automatically written to the corresponding file. For data sample, it should be empty.
|- feature_selection.py   * feature_selection
|- global_params.py       * global_params
|- model_ANN.py           * four ANN model
|- model_LightGBM.py      * LGBM model
|- model_xgboost.py       * xgboost model
|- preprocess.py          * Process time series. Data cleaning and preprocessing.
|- read.py                * support io for the dataset
|- utils.py               * Some helper functions are written in this file to facilitate code reuse
|- stacking.py            * model ensemble(stacking)
```

## How to run code?
**Since code commits can only be 5MB, we used a very, very small sample dataset and removed some model(When parameter sample is True). To keep the code running, don't delete any empty folders.**

If using sample dataset: 

```
1. set ANN_params['model'] = 'Logistic' | 'ANN' | 'ANN_Norm' | 'ANN_norm_res_dropout' and run model_ANN.py
2. run model_LightGBM.py model_xgboost.py
3. set ens_params['models'] = [models your have run and get rusults] and run stacking.py
4. set preprocess_method in global_params.py.
5. If your want to choose hyperparameters, enter model_*.py to modify, it is always at the top of the code.
```

If using total dataset: 

```
1. You can choose to run feature_selection.py and set feature_selection and feature_selection_method in global_params.py.
2. If you want to get preditions which is submited to kaggle, set submit=True in global_params.py.
3. Others are same as 1,2,3,4,5 in sample dataset.
```

