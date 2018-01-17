import pandas as pd
import numpy as np
import timeit
from sklearn.linear_model import LinearRegression



#===============================================================================


def calc_window_stats(sales, start, window, groupby, aggregates, value='unit_sales', colnames=None):

    last_date = pd.to_datetime(start) - pd.Timedelta(days=1)
    first_date = pd.to_datetime(start) - pd.Timedelta(days=window)
    df = sales[(sales.date>=first_date) & (sales.date<=last_date)]

    df2 = df.groupby(groupby)[value].agg(aggregates)
    if colnames is not None:
        df2.columns = colnames
    return df2.reset_index()


def make_names(suffix, agg):
        return list(map(lambda x: x+suffix, agg))


def sales_by_store_item(sales, start, agg=['mean'], value='unit_sales'):

    print('adding sales by store/item...')
    data = calc_window_stats(sales, start, window=1, groupby=['store_nbr', 'item_nbr'],
                aggregates=['last'], value=value,
                colnames=make_names('_1', agg))
    print('window 1 added')

    for window in [3,7,14,28,60,90,180,365]:
        df = calc_window_stats(sales, start, window=window, groupby=['store_nbr', 'item_nbr'],
                    aggregates=agg, value=value,
                    colnames=make_names('_%d' % window, agg))
        data = pd.merge(data, df)
        print('window %d added' % window)

    return data


def sales_by_store_item_dow(sales, start, agg=['mean'], value='unit_sales'):

    print('adding sales by store/item/dow...')
    data = calc_window_stats(sales, start, window=7, groupby=['store_nbr', 'item_nbr', 'dow'],
                aggregates=['last'], value=value,
                colnames=make_names('_7_dow', agg))
    data = data.set_index(['store_nbr', 'item_nbr', 'dow']).unstack()
    data.columns = data.columns.get_level_values(0) + data.columns.get_level_values(1).astype(str)
    data = data.reset_index()
    print('window 7 added')

    for window in [14,28,28*2,28*3,28*6,364]:
        df = calc_window_stats(sales, start, window=window, groupby=['store_nbr', 'item_nbr', 'dow'],
                    aggregates=agg, value=value,
                    colnames=make_names('_%d_dow' % window, agg))
        df = df.set_index(['store_nbr', 'item_nbr', 'dow']).unstack()
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1).astype(str)
        df = df.reset_index()
        data = pd.merge(data, df)
        print('window %d added' % window)

    return data


def sales_by_store_item_day(sales, start, agg=['mean'], value='unit_sales'):

    print('adding sales by store/item/day...')
    data_list = []

    for window in [365]:
        df = calc_window_stats(sales, start, window=window, groupby=['store_nbr', 'item_nbr', 'day'],
                    aggregates=agg, value=value,
                    colnames=make_names('_%d_day' % window, agg))
        df = df.set_index(['store_nbr', 'item_nbr', 'day']).unstack()
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1).astype(str)
        df = df.reset_index()
        data_list.append(df)
        print('window %d added' % window)

    if len(data_list)>1:
        data = data_list[0]
        for df in data_list[1:]:
            data = pd.merge(data, df)
    else:
        data = data_list[0]

    return data


def zerodays_by_store_item(sales, start):

    print('adding zerodays by store/item...')
    data_list = []

    for window in [7, 28, 90, 365]:
        df = calc_window_stats(sales, start, window=window, groupby=['store_nbr', 'item_nbr'],
                    aggregates=['mean'], value='zero_day',
                    colnames=['mean_zerodays_%d' % window])
        data_list.append(df)
        print('window %d added' % window)

    data = data_list[0]
    for df in data_list[1:]:
        data = pd.merge(data, df)

    return data


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - pd.Timedelta(days=minus), periods=periods, freq=freq)]

def promo_by_store_item(promo, start):

    print('adding promo by store/item...')

    X = pd.DataFrame({
        "promo_mean_7": get_timespan(promo, start, 7, 7).mean(axis=1).values,
        "promo_mean_14": get_timespan(promo, start, 14, 14).mean(axis=1).values,
        "promo_mean_28": get_timespan(promo, start, 28, 28).mean(axis=1).values,
        "promo_mean_60": get_timespan(promo, start, 60, 60).mean(axis=1).values,
        "promo_mean_90": get_timespan(promo, start, 90, 90).mean(axis=1).values,
        "promo_mean_180": get_timespan(promo, start, 180, 180).mean(axis=1).values,
        "promo_mean_365": get_timespan(promo, start, 365, 365).mean(axis=1).values
    },
        index=promo.index)

    for i in range(16):
        X["promo_%d" % i] = promo[start + pd.Timedelta(days=i)].values

    return X.reset_index()


def sales_by_store_item_category(sales, start, category='family', value='unit_sales'):

    print('adding sales by store/%s...' % category)
    data_list = []

    for window in [7, 28]:
        df = calc_window_stats(sales, start, window=window, groupby=['store_nbr', category],
                    aggregates=['mean'], value=value,
                    colnames=['mean_store_%s_%d' % (category, window)])
        df = pd.merge(pairs[['store_nbr', 'item_nbr', category]], df)
        df.drop(category, axis=1, inplace=True)
        df = df.sort_values(['store_nbr', 'item_nbr'])
        data_list.append(df)
        print('window %d added' % window)

    data = data_list[0]
    for df in data_list[1:]:
        data = pd.merge(data, df)

    return data


def sales_by_item_store_category(sales, start, category='city', value='unit_sales'):

    print('adding sales by item/%s...' % category)
    data_list = []

    for window in [7, 28]:
        df = calc_window_stats(sales, start, window=window, groupby=['item_nbr', category],
                    aggregates=['mean'], value=value,
                    colnames=['mean_item_%s_%d' % (category, window)])
        df = pd.merge(pairs[['store_nbr', 'item_nbr', category]], df)
        df.drop(category, axis=1, inplace=True)
        df = df.sort_values(['store_nbr', 'item_nbr'])
        data_list.append(df)
        print('window %d added' % window)

    data = data_list[0]
    for df in data_list[1:]:
        data = pd.merge(data, df)

    return data


def sales_by_item(sales, start, value='unit_sales'):

    print('adding sales by item...')
    data_list = []

    for window in [7, 28]:
        df = calc_window_stats(sales, start, window=window, groupby=['item_nbr'],
                    aggregates=['mean'], value=value,
                    colnames=['mean_item_%d' % ( window)])
        df = pd.merge(pairs[['store_nbr', 'item_nbr']], df)
        df = df.sort_values(['store_nbr', 'item_nbr'])
        data_list.append(df)
        print('window %d added' % window)

    data = data_list[0]
    for df in data_list[1:]:
        data = pd.merge(data, df)

    return data


def zerodays_by_store(sales, start):

    print('adding zerodays by store...')
    data_list = []

    for window in [7, 28]:
        df = calc_window_stats(sales, start, window=window, groupby=['store_nbr'],
                    aggregates=['mean'], value='zero_day',
                    colnames=['mean_store_zerodays_%d' % window])
        df = pd.merge(pairs[['store_nbr', 'item_nbr']], df)
        df = df.sort_values(['store_nbr', 'item_nbr'])
        data_list.append(df)
        print('window %d added' % window)

    data = data_list[0]
    for df in data_list[1:]:
        data = pd.merge(data, df)

    return data


def regression(df, value='unit_sales', colname='regression', pred_len=16):
    lr = LinearRegression()
    ts = df[value].reset_index(drop=True)
    lr.fit(ts.index.values.reshape(-1, 1), ts.values)
    pred_index = np.array(range(ts.index.max()+1, ts.index.max()+1+pred_len))
    return pd.Series(lr.predict(pred_index.reshape(-1, 1)), index=[colname + '_%d' % x for x in range(pred_len)])


def calc_regression(sales, start, window, groupby, value='unit_sales', colname='regression', pred_len=16):

    last_date = pd.to_datetime(start) - pd.Timedelta(days=1)
    first_date = pd.to_datetime(start) - pd.Timedelta(days=window)
    df = sales[(sales.date>=first_date) & (sales.date<=last_date)]

    df2 = df.groupby(groupby).apply(regression, value=value, colname=(colname+'%d' % window), pred_len=pred_len)

    return df2.reset_index()


def regression_by_store_item(sales, start, value='unit_sales'):

    print('adding regression by store/item...')
    data_list = []

    for window in [50,100]:
        df = calc_regression(sales, start, window=window, groupby=['store_nbr', 'item_nbr'],
                             value=value, colname='regression')
        data_list.append(df)
        print('window %d added' % window)

    data = data_list[0]
    for df in data_list[1:]:
        data = pd.merge(data, df)

    return data


def regression_by_store_item_dow(sales, start, value='unit_sales'):

    print('adding regression by store/item/dow...')
    data_list = []

    for window in [7*20,7*52]:
        df = calc_regression(sales, start, window=window, groupby=['store_nbr', 'item_nbr', 'dow'],
                             value=value, colname='regression', pred_len=2)
        df = df.set_index(['store_nbr', 'item_nbr', 'dow']).unstack()
        df.columns = df.columns.get_level_values(0) + '_dow_' + df.columns.get_level_values(1).astype(str)
        df = df.reset_index()
        data_list.append(df)
        print('window %d added' % window)

    data = data_list[0]
    for df in data_list[1:]:
        data = pd.merge(data, df)

    return data


def target(sales, start):
    y = sales[pd.date_range(start, periods=16)]
    return y


def calculate_features(sales, start, sales_value):

    start_time = timeit.default_timer()
    data = sales_by_store_item(sales, start, agg=['mean'], value=sales_value)
    data['mean_diff_7_28'] = data.mean_7 - data.mean_28
    data['mean_diff_14_60'] = data.mean_14 - data.mean_60
    data['mean_diff_28_90'] = data.mean_28 - data.mean_90
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_store_item_dow(sales, start, agg=['mean'], value=sales_value)
    data = pd.merge(data, df)
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_store_item_day(sales, start, agg=['mean'], value=sales_value)
    data = pd.merge(data, df)
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = zerodays_by_store_item(sales, start)
    data = pd.merge(data, df)
    data['mean_zerodays_diff_7_28'] = data.mean_zerodays_7 - data.mean_zerodays_28
    data['mean_zerodays_diff_28_90'] = data.mean_zerodays_28 - data.mean_zerodays_90
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = promo_by_store_item(promo, start)
    data = pd.merge(data, df)
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_store_item_category(sales, start, category='family', value=sales_value)
    data = pd.merge(data, df)
    data['mean_store_family_diff_7_28'] = data.mean_store_family_7 - data.mean_store_family_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_store_item_category(sales, start, category='class', value=sales_value)
    data = pd.merge(data, df)
    data['mean_store_class_diff_7_28'] = data.mean_store_class_7 - data.mean_store_class_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_item_store_category(sales, start, category='city', value=sales_value)
    data = pd.merge(data, df)
    data['mean_item_city_diff_7_28'] = data.mean_item_city_7 - data.mean_item_city_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_item_store_category(sales, start, category='state', value=sales_value)
    data = pd.merge(data, df)
    data['mean_item_state_diff_7_28'] = data.mean_item_state_7 - data.mean_item_state_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_item_store_category(sales, start, category='type', value=sales_value)
    data = pd.merge(data, df)
    data['mean_item_type_diff_7_28'] = data.mean_item_type_7 - data.mean_item_type_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_item_store_category(sales, start, category='cluster', value=sales_value)
    data = pd.merge(data, df)
    data['mean_item_cluster_diff_7_28'] = data.mean_item_cluster_7 - data.mean_item_cluster_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = sales_by_item(sales, start,  value=sales_value)
    data = pd.merge(data, df)
    data['mean_item_diff_7_28'] = data.mean_item_7 - data.mean_item_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = zerodays_by_store(sales, start)
    data = pd.merge(data, df)
    data['mean_store_zerodays_diff_7_28'] = data.mean_store_zerodays_7 - data.mean_store_zerodays_28
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

    df = regression_by_store_item(sales, start, value=sales_value)
    data = pd.merge(data, df)
    print(data.shape)
    print('time elapsed', timeit.default_timer()-start_time)

# #   very long computations
#     df = regression_by_store_item_dow(sales, start, value=sales_value)
#     data = pd.merge(data, df)
#     print(data.shape)
#     print('time elapsed', timeit.default_timer()-start_time)

    return data



#===============================================================================


train = pd.read_csv('data/train.csv', usecols=[1, 2, 3, 4, 5],
                    dtype={'store_nbr': 'int8', 'item_nbr': 'int32', 'unit_sales': 'float32'},
                    #skiprows=range(1, 66458909)  # 2016-01-01
                 )
test = pd.read_csv('data/test.csv',
                    dtype={'id': 'int32', 'store_nbr': 'int8', 'item_nbr': 'int32',
                            'unit_sales': 'float32'})
items = pd.read_csv("data/items.csv",
                    type={'item_nbr': 'int32', 'class': 'int16', 'perishable': 'int8'})
stores = pd.read_csv("data/stores.csv",
                    dtype={'store_nbr': 'int8', 'cluster': 'int8'})

train.onpromotion = train.onpromotion.fillna(-1).astype(int)
test.onpromotion = test.onpromotion.fillna(-1).astype(int)


first_date = '2014-04-01'
train = train[train.date>=first_date]

train.date = pd.to_datetime(train.date, format='%Y-%m-%d')
test.date = pd.to_datetime(test.date, format='%Y-%m-%d')

train.unit_sales = train.unit_sales.clip(lower=0)


# wide promo table
promo_train = train.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack().fillna(0)
promo_train.columns = promo_train.columns.get_level_values(1)
promo_test = test.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack().fillna(0)
promo_test.columns = promo_test.columns.get_level_values(1)
promo_test = promo_test.reindex(promo_train.index).fillna(0)
promo = pd.concat([promo_train, promo_test], axis=1)
del promo_train, promo_test


# wide sales table
sales_wide = train.set_index(['store_nbr', 'item_nbr', 'date']).unit_sales.unstack().fillna(0)
print(sales_wide.shape)


# add days with zero sales
sales = sales_wide.stack().rename('unit_sales').reset_index()
print(sales.shape)


# 2016-12-25 is absent from train, need to fill it!
sales_wide = sales_wide.reindex_axis(pd.date_range(first_date, '2017-08-15'), axis=1).fillna(0)
promo = promo.reindex_axis(pd.date_range(first_date, '2017-08-31'), axis=1).fillna(0)
print(sales_wide.shape, promo.shape)


sales['dow'] = sales.date.dt.dayofweek
sales['day'] = sales.date.dt.day
sales['zero_day'] = (sales.unit_sales==0).astype('int8')
sales['unit_sales_log'] = np.log(1+sales.unit_sales)

sales = sales.astype({'store_nbr': 'int8', 'item_nbr': 'int32', 'dow': 'int8', 'day': 'int8', 'zero_day': 'int8'})

sales = pd.merge(sales, items.drop('perishable', axis=1))
print(sales.shape)
sales = pd.merge(sales, stores)
print(sales.shape)


pairs = sales[['store_nbr', 'item_nbr']].drop_duplicates()
pairs = pd.merge(pairs, items)
pairs = pd.merge(pairs, stores)
print(pairs.shape)


t = pd.to_datetime('2017-07-05')
train_start = []
for i in range(25):
    delta = pd.Timedelta(days=7 * i)
    train_start.append(t-delta)
print(train_start)

val_start = pd.to_datetime('2017-07-26')
test_start = pd.to_datetime('2017-08-16')


val_data = calculate_features(sales, val_start, sales_value='unit_sales_log')
y_val = target(sales_wide, val_start)
val_data.to_csv('features/val_data.csv', index=False)
y_val.to_csv('target/y_val.csv')

test_data = calculate_features(sales, test_start, sales_value='unit_sales_log')
test_data.to_csv('features/test_data3.csv', index=False)


for start in train_start:

    print(start)

    train_data = calculate_features(sales, start, sales_value='unit_sales_log')
    y_train = target(sales_wide, start)

    train_data.to_csv('features/train_data_%s.csv' % start.strftime('%Y-%m-%d'), index=False)
    y_train.to_csv('target/y_train_%s.csv' % start.strftime('%Y-%m-%d'))
