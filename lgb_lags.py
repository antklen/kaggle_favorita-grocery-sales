import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import timeit
import gc



def nwrmsle(yval, ypred, weights=None):
    return np.sqrt(mean_squared_error(np.log(1+yval), np.log(1+ypred), sample_weight=weights))


def lag_features(df0, lag, y0=None):

    test_range = pd.date_range('2017-08-16', '2017-08-31')

    if y0 is not None:
        date = pd.to_datetime(y0.columns[lag])
    else:
        date = test_range[lag]
    day = date.day
    dow = date.dayofweek

    df = df0[['mean_1', 'mean_3', 'mean_7', 'mean_14', 'mean_28', 'mean_60', 'mean_90', 'mean_180', 'mean_365',
                   'mean_diff_7_28', 'mean_diff_14_60', 'mean_diff_28_90',
                   'mean_zerodays_7', 'mean_zerodays_28', 'mean_zerodays_90', 'mean_zerodays_365',
                   'mean_zerodays_diff_7_28', 'mean_zerodays_diff_28_90',
                   'promo_mean_28', 'promo_mean_365', 'promo_mean_7', 'promo_mean_90',
                   'mean_store_family_7', 'mean_store_family_28', 'mean_store_family_diff_7_28',
                   'mean_store_class_7', 'mean_store_class_28', 'mean_store_class_diff_7_28',
                   'mean_item_city_7', 'mean_item_city_28', 'mean_item_city_diff_7_28',
                   'mean_item_state_7', 'mean_item_state_28', 'mean_item_state_diff_7_28',
                   'mean_item_type_7', 'mean_item_type_28', 'mean_item_type_diff_7_28',
                   'mean_item_cluster_7', 'mean_item_cluster_28', 'mean_item_cluster_diff_7_28',
                   'mean_item_7', 'mean_item_28', 'mean_item_diff_7_28',
                   'mean_store_zerodays_7', 'mean_store_zerodays_28', 'mean_store_zerodays_diff_7_28']]

    df['mean_dow_7'] = df0['mean_7_dow%d' % dow]
    df['mean_dow_14'] = df0['mean_14_dow%d' % dow]
    df['mean_dow_28'] = df0['mean_28_dow%d' % dow]
    df['mean_dow_56'] = df0['mean_56_dow%d' % dow]
    df['mean_dow_84'] = df0['mean_84_dow%d' % dow]
    df['mean_dow_168'] = df0['mean_168_dow%d' % dow]
    df['mean_dow_364'] = df0['mean_364_dow%d' % dow]

    df['mean_day_365'] = df0['mean_365_day%d' % day]

    df['regression_50'] = df0['regression50_%d' % lag]
    df['regression_100'] = df0['regression100_%d' % lag]

    df['promo'] = df0['promo_%d' % lag]
    df['promo_mean'] = df0[['promo_0', 'promo_1', 'promo_2', 'promo_3', 'promo_4', 'promo_5', 'promo_6', 'promo_7',
            'promo_8', 'promo_9', 'promo_10', 'promo_11', 'promo_12', 'promo_13', 'promo_14', 'promo_15']].mean(axis=1)

    df['family'] = items2['family'].values
    df['class'] = items2['class'].values
    df['perishable'] = items2['perishable'].values
    df['city'] = stores2['city'].values
    df['state'] = stores2['state'].values
    df['type'] = stores2['type'].values
    df['cluster'] = stores2['cluster'].values

    df['dow'] = dow
    df['day'] = day

    df = df.reset_index()

    if y0 is not None:
        y_i = y0.iloc[:,lag].rename('y').to_frame()
        y_i['date'] = date
        y_i = y_i.reset_index().set_index(['store_nbr', 'item_nbr', 'date']).squeeze()
        return df, y_i
    else:
        return df



# prepare data

val0 = pd.read_csv('features/val_data.csv', index_col=[0,1])
test0 = pd.read_csv('features/test_data.csv', index_col=[0,1])
yval0 = pd.read_csv('target/y_val.csv', index_col=[0,1])

t = pd.to_datetime('2017-07-05')
train_start = []
for i in range(25):
    delta = pd.Timedelta(days=7 * i)
    train_start.append((t-delta).strftime('%Y-%m-%d'))
print(train_start)

train0 = []
ytrain0 = []
for start in train_start:
    print(start)
    train0.append(pd.read_csv('features/train_data_%s.csv' % start, index_col=[0,1]))
    ytrain0.append(pd.read_csv('target/y_train_%s.csv' % start, index_col=[0,1]))


items = pd.read_csv('data/items.csv')
stores = pd.read_csv('data/stores.csv')

le = LabelEncoder()
items.family = le.fit_transform(items.family)
stores.city = le.fit_transform(stores.city)
stores.state = le.fit_transform(stores.state)
stores.type = le.fit_transform(stores.type)

items2 = items.set_index('item_nbr').reindex(val0.index.get_level_values(1))
print(items2.shape)
stores2 = stores.set_index('store_nbr').reindex(val0.index.get_level_values(0))
print(stores2.shape)



# train model


pred = []
test_pred = []

for i in range(16):

    print("=" * 50)
    print("Step %d" % (i))
    print("=" * 50)
    start_timer = timeit.default_timer()


    val, yval = lag_features(val0, i, yval0)
    test = lag_features(test0, i)

    train = []
    ytrain = []
    for j in range(len(train0)):
        tr, ytr = lag_features(train0[j], i, ytrain0[j])
        train.append(tr)
        ytrain.append(ytr)
    train = pd.concat(train)
    ytrain = pd.concat(ytrain)

    gc.collect()


    clf = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.05, num_leaves=150, min_data_in_leaf=200,
                        subsample=0.8, colsample_bytree=0.3, random_state=42, n_jobs=-1)

    clf.fit(train, np.log(1+ytrain), eval_set=[(val, np.log(1+yval))], early_stopping_rounds=100,
                eval_metric='rmse', sample_weight=(train.perishable.values*0.25+1), verbose=50)

    display(pd.Series(clf.booster().feature_importance(importance_type='gain'),
                      index=train.columns).sort_values(ascending=False).head(20))

    pred.append(np.exp(clf.predict(val, num_iteration=clf.best_iteration_))-1)
    test_pred.append(np.exp(clf.predict(test, num_iteration=clf.best_iteration_))-1)

    print('nwrmsle %.5f' % nwrmsle(yval.values, pred[-1], weights=val.perishable.values*0.25+1))
    print('time elapsed', timeit.default_timer()-start_timer)


ypred = np.array(pred).transpose()
y_test = np.array(test_pred).transpose()


print('Validation error')
print('rmsle %.5f' % nwrmsle(yval0.values, ypred))
print('nwrmsle %.5f' % nwrmsle(yval0.values, ypred, weights=items2.perishable.values*0.25+1))
print('nwrmsle-first5 %.5f' % nwrmsle(yval0.values[:,:5], ypred[:,:5], weights=items2.perishable.values*0.25+1))
print('nwrmsle-last11 %.5f' % nwrmsle(yval0.values[:,5:], ypred[:,5:], weights=items2.perishable.values*0.25+1))



# make submission

df_test = pd.read_csv("data/test.csv", usecols=[0, 1, 2, 3], parse_dates=["date"]). \
                    set_index(['store_nbr', 'item_nbr', 'date'])
df_preds = pd.DataFrame(y_test, index=test0.index,
                columns=pd.date_range("2017-08-16", periods=16)).stack().to_frame('unit_sales')
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
submission = df_test.join(df_preds, how="left").fillna(0)
submission.unit_sales = submission.unit_sales.clip(lower=0)

submission.to_csv('submissions/submission.csv', float_format='%.6f', index=False)
