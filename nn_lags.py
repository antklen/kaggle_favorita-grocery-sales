import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import timeit
import gc

import keras
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model


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


def nn_model(input_num_shape):

    input_num = Input(shape=(input_num_shape,), dtype='float32', name='input_num')
    input_store = Input(shape=(1,), dtype='int32', name='input_store')
    input_item = Input(shape=(1,), dtype='int32', name='input_item')
    input_family = Input(shape=(1,), dtype='int32', name='input_family')
    input_city = Input(shape=(1,), dtype='int32', name='input_city')
    input_state = Input(shape=(1,), dtype='int32', name='input_state')
    input_type = Input(shape=(1,), dtype='int32', name='input_type')
    input_cluster = Input(shape=(1,), dtype='int32', name='input_cluster')
    input_dow = Input(shape=(1,), dtype='int32', name='input_dow')
    input_day = Input(shape=(1,), dtype='int32', name='input_day')

    embedding_store = Embedding(input_dim=val0.reset_index()['store_nbr'].nunique(), output_dim=5, input_length=1)(input_store)
    embedding_store = Flatten()(embedding_store)

    embedding_item = Embedding(input_dim=val0.reset_index()['item_nbr'].nunique(), output_dim=10, input_length=1)(input_item)
    embedding_item = Flatten()(embedding_item)

    embedding_family = Embedding(input_dim=items['family'].nunique(), output_dim=5, input_length=1)(input_family)
    embedding_family = Flatten()(embedding_family)

    embedding_city = Embedding(input_dim=stores['city'].nunique(), output_dim=5, input_length=1)(input_city)
    embedding_city = Flatten()(embedding_city)

    embedding_state = Embedding(input_dim=stores['state'].nunique(), output_dim=5, input_length=1)(input_state)
    embedding_state = Flatten()(embedding_state)

    embedding_type = Embedding(input_dim=stores['type'].nunique(), output_dim=3, input_length=1)(input_type)
    embedding_type = Flatten()(embedding_type)

    embedding_cluster = Embedding(input_dim=stores['cluster'].nunique(), output_dim=5, input_length=1)(input_cluster)
    embedding_cluster = Flatten()(embedding_cluster)

    embedding_dow = Embedding(input_dim=7, output_dim=5, input_length=1)(input_dow)
    embedding_dow = Flatten()(embedding_dow)

    embedding_day = Embedding(input_dim=31, output_dim=5, input_length=1)(input_day)
    embedding_day = Flatten()(embedding_day)


    features = [input_num, embedding_store, embedding_item, embedding_family, embedding_city,
               embedding_state, embedding_type, embedding_cluster, embedding_dow, embedding_day]
    net = concatenate(features)
    net = Dense(1000, kernel_initializer = 'he_normal', activation='relu')(net)
    net = Dense(500, kernel_initializer = 'he_normal', activation='relu')(net)
    net = Dense(1, kernel_initializer = 'he_normal', activation='linear')(net)

    inputs = [input_num, input_store, input_item, input_family, input_city,
             input_state, input_type, input_cluster, input_dow, input_day]

    model = Model(inputs=inputs, outputs=[net])
    model.compile(loss='mse', optimizer='adam')

    return(model)



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

    train_set = [train.drop(['store_nbr', 'item_nbr', 'family', 'class', 'city', 'state', 'type', 'cluster',
                             'dow', 'day'], axis=1).values,
                train.store_nbr.values, train.item_nbr.values,
                train.family.values,
                train.city.values, train.state.values, train.type.values, train.cluster.values,
                train.dow.values, train.day.values]
    val_set = [val.drop(['store_nbr', 'item_nbr', 'family', 'class', 'city', 'state', 'type', 'cluster',
                             'dow', 'day'], axis=1).values,
                val.store_nbr.values, val.item_nbr.values,
                val.family.values,
                val.city.values, val.state.values, val.type.values, val.cluster.values,
                val.dow.values, val.day.values]
    test_set = [test.drop(['store_nbr', 'item_nbr', 'family', 'class', 'city', 'state', 'type', 'cluster',
                             'dow', 'day'], axis=1).values,
                test.store_nbr.values, test.item_nbr.values,
                test.family.values,
                test.city.values, test.state.values, test.type.values, test.cluster.values,
                test.dow.values, test.day.values]

    gc.collect()


    model = nn_model(train.shape[1]-10)

    earlyStopping=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath="tmp/weights.h5", verbose=0, save_best_only=True,
                               save_weights_only=True)

    model.fit(train_set, np.log(1+ytrain.values),
            validation_data=(val_set, np.log(1+yval.values)),
            epochs=15, batch_size=512, verbose=1, callbacks=[earlyStopping, checkpointer])

    model.load_weights('tmp/weights.h5')
    pred.append(np.exp(model.predict(val_set))-1)
    test_pred.append(np.exp(model.predict(test_set))-1)

    print('nwrmsle %.5f' % nwrmsle(yval.values, pred[-1], weights=val.perishable.values*0.25+1))
    print('time elapsed', timeit.default_timer()-start_timer)


ypred = np.array(pred).transpose().squeeze()
y_test = np.array(test_pred).transpose().squeeze()


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
