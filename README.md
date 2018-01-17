## Corporación Favorita Grocery Sales Forecasting

Main parts of 12th place solution of kaggle Corporación Favorita Grocery Sales Forecasting competition.

### files description

feature_engineering.py - calculating and saving engineered features

lgb_general.py - general LightGBM model, trained on data for all lags simultaneously

lgb_lags.py - LightGBM model, trained for each lag separately

nn_lags.py - neural network model, trained for each lag separately
