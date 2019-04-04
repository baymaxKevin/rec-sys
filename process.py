# coding=utf-8
import pandas as pd

import json
import os
from compiler.ast import flatten
prefix = os.path.abspath(os.path.dirname(os.getcwd()))
data = pd.read_csv(prefix + '/data/data.csv', sep=',')
del data['userGest']


# 拆分物品id，类型和关键词
def split_item_cate_key(str, target):
    temp_list = []
    list = str.split(',')
    for ele in list:
        item, type, cate, keyword = ele.split('_')
        if type == '2':  # 话题先不处理
            if (target == 1) & (item != ''):
                temp_list.append(float(item))
            elif (target == 2) & (cate != ''):
                temp_list.append(float(cate))
            elif (target == 3) & (keyword != ''):
                temp_list.append(float(keyword))
            else:
                temp_list.append(-1.0)
        else:
            continue
    return temp_list


data['userItemHistory'] = data['userHList'].apply(lambda x: split_item_cate_key(x, target=1))
data['userCateHistory'] = data['userHList'].apply(lambda x: split_item_cate_key(x, target=2))
data['userKeywordHistory'] = data['userHList'].apply(lambda x: split_item_cate_key(x, target=3))
del data['userHList']


# label编码
def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    print(col_name)
    return m, key, len(m)


def user_stable(columns, match):
    list1 = []
    for ele in columns:
        if ele.find(match) > -1:
            list1.append(ele)
    return list1


def dump_dict(dict, filepath):
    jsObj = json.dumps(dict)
    fileObject = open(filepath, 'w')
    fileObject.write(jsObj)
    fileObject.close()


# 将所有属性按位置编码，当做一个大属性
def merge_all_feature(features, data, dict):
    for col in features:
        data[col] = data[col].map(lambda x: x + dict[col])


# 用户固定属性集合18个
user_stable_feature_list = user_stable(data.columns.tolist(), match='user')
user_stable_feature_list.remove('userItemHistory')
user_stable_feature_list.remove('userCateHistory')
user_stable_feature_list.remove('userKeywordHistory')
user_dict = {}  # 记录每个特征的值个数
user_position_dict = {}
position = 0
for col in user_stable_feature_list:
    _, _, length = build_map(data, col)
    user_dict[col] = length
    user_position_dict[col] = position
    position = position + length
user_dict['user_all_length'] = position
dump_dict(user_dict, prefix + '/data/user_dict.txt')
merge_all_feature(features=user_stable_feature_list, data=data, dict=user_position_dict)

# 物品固定属性集合21个
item_stable_feature_list = user_stable(data.columns.tolist(), match='item')
item_stable_feature_list.remove('itemId')
item_stable_feature_list.remove('itemTtP')
item_stable_feature_list.remove('itemKtW')
item_dict = {}  # 记录每个特征的值个数
item_position_dict = {}
position = 0
for col in item_stable_feature_list:
    _, _, length = build_map(data, col)
    item_dict[col] = length
    item_position_dict[col] = position
    position = position + length
item_dict['item_all_length'] = position
dump_dict(item_dict, prefix + '/data/item_dict.txt')
merge_all_feature(features=item_stable_feature_list, data=data, dict=item_position_dict)


def list_map(x, m_dict):
    l = []
    for ele in x:
        l.append(m_dict[ele])
    return l


def merge_item(x, z, data, history, single):
    list1 = flatten(x)
    list1 = list1 + z
    key = list(set(list1))
    key.sort()
    m_dict = dict(zip(key, range(len(key))))
    data[single] = data[single].map(lambda x: m_dict[x])
    data[history] = data[history].apply(lambda x: list_map(x, m_dict))
    return len(m_dict)


# 记录item_count, cate_count, keyword_count, user_features_num, item_features_num, user_features_dim, item_features_dim,
count_dict = {}
count_dict['user_features_num'] = len(user_stable_feature_list)
count_dict['item_features_num'] = len(item_stable_feature_list)
count_dict['user_features_dim'] = user_dict['user_all_length']
count_dict['item_features_dim'] = item_dict['item_all_length']
# 处理item，种类，关键词
for history, single, key in [('userItemHistory', 'itemId', 'item_count'), ('userCateHistory', 'itemTtP', 'cate_count'),
                             ('userKeywordHistory', 'itemKtW', 'keyword_count')]:
    count = merge_item(data[history].tolist(), data[single].tolist(), data=data, history=history, single=single)
    count_dict[key] = count
dump_dict(count_dict, prefix + '/data/count_dict.txt')


# 按此格式组织数据
data['item_feature'] = data[item_stable_feature_list].apply(lambda x: x.values, axis=1)
data['user_feature'] = data[user_stable_feature_list].apply(lambda x: x.values, axis=1)

data.drop(item_stable_feature_list, axis=1, inplace=True)
data.drop(user_stable_feature_list, axis=1, inplace=True)

data = data[['label', 'user_feature', 'item_feature', 'itemId', 'itemTtP', 'itemKtW', 'userItemHistory',
             'userCateHistory', 'userKeywordHistory']]
data.to_pickle(prefix + '/data/train_set.pkl')
