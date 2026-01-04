import pandas as pd
import numpy as np
import os
import re
from data_features_preview import *
import json

# 数据预处理
# 在初审数据时我们发现一部分的特征缺失
# 有许多string类型的特征需要量化
# 需要添加一些必要特征
# 要将所有数据处理为能够建模的数据


# 想法是
# 1. 删除id/重复列/...
# 2. 处理null数据,将其统一删除或变为某个什么东西
# 3. 量化brand,model等数据
# 4. 修改一些特征，例如将提车年份feature改成用车时限等
# 5. 对于每个特征，出现次数过少的值做统一化处理



#之后ui部分用到的数据
class UsefulFeature():
    def __init__(self, brand_model_dict, dic):
        self.brand_model_dict = brand_model_dict
        self.dic = dic



def add_age_features(df):
    year = 2025
    df['Vehicle_Age'] = year - df['model_year']
    # 如下部分的特征选取想法来自kaggle用户HOON0303
    df['Mileage_per_Year'] = df['milage'] / df['Vehicle_Age']
    df['milage_with_age'] = df.groupby('Vehicle_Age')['milage'].transform('mean')
    df['Mileage_per_Year_with_age'] = df.groupby('Vehicle_Age')['Mileage_per_Year'].transform('mean')
    # 输出每个 Vehicle_Age 对应的 milage 的均值
    print(df.groupby('Vehicle_Age')['milage'].mean())

    # 输出每个 Vehicle_Age 对应的 Mileage_per_Year 的均值
    print(df.groupby('Vehicle_Age')['Mileage_per_Year'].mean())
    df.drop(['model_year'], axis=1, inplace=True)


def add_engine_features(df):
    def horsepower(engine):
        try:
            return float(engine.split('HP')[0])
        except:
            return None

    def engine_size(engine):
        try:
            return float(engine.split(' ')[1].replace('L', ''))
        except:
            return None

    df['Horsepower'] = df['engine'].apply(horsepower)
    df['Engine_Size'] = df['engine'].apply(engine_size)
    df['Power_to_Weight_Ratio'] = df['Horsepower'] / df['Engine_Size']
    df['is_electric'] = df['engine'].apply(lambda x: 1 if 'electric' in x.lower() else 0)
    df.drop(['engine'], axis=1, inplace=True)







# 将缺失值补全，注意到只有fuel_type,accident和clean_title三项有缺失
# 后续操作又出现了Horsepower,Engine_Size和Power_to_Weight_Ratio的缺失,用平均数值补全
def fill_missing_values(df):
    # 将fuel_type,accident和clean_title三项的缺失值补全
    df['fuel_type'].fillna('unknown or other (like electricity)', inplace=True)
    df['accident'].fillna('unknown or unwilling to dispose', inplace=True)
    df['clean_title'].fillna('unknown', inplace=True)
    # 将Horsepower,Engine_Size和Power_to_Weight_Ratio的缺失值补全
    df['Horsepower'].fillna(df['Horsepower'].mean(), inplace=True)
    df['Engine_Size'].fillna(df['Engine_Size'].mean(), inplace=True)
    df['Power_to_Weight_Ratio'].fillna(df['Power_to_Weight_Ratio'].mean(), inplace=True)

# 删除id
def drop_id(df):

    df.drop(['id'], axis=1, inplace=True)



# 删除重复数据
def drop_duplicates(df):

    df.drop_duplicates(inplace=True)




# 将model中少于20个数据的改为other
# 将ext_col和int_col中分别少于240/480个数据的（均值的40%）改为other
# 将transmission中少于400个数据的（均值的10%）改为other
def update(df):

    model_counts = df['model'].value_counts()
    # 将model中少于10个数据的（均值的10%）改为other
    df['model'] = df['model'].apply(lambda x: x if model_counts[x] >= 20 else 'other')

    # 统计ext_col和int_col中每个值的数量
    ext_col_counts ,int_col_counts= df['ext_col'].value_counts(),df['int_col'].value_counts()
    # 将ext_col和int_col中分别少于60/120个数据的（均值的10%）改为other
    df['ext_col'], df['int_col']= df['ext_col'].apply(lambda x: x if ext_col_counts[x] >= 240 else 'other'),df['int_col'].apply(lambda x: x if int_col_counts[x] >= 480 else 'other')
    # 将ext_col和int_col中为-的改为other
    df['ext_col'] = df['ext_col'].apply(lambda x: 'other' if x == '–' else x)
    df['int_col'] = df['int_col'].apply(lambda x: 'other' if x == '–' else x)


    # 将transmission中少于400个数据的（均值的10%）改为other
    transmission_counts = df['transmission'].value_counts()
    df['transmission'] = df['transmission'].apply(lambda x: x if transmission_counts[x] >= 400 else 'other')

    # 将clean_title的Yes赋为1，No赋为0
    df['clean_title'] = df['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)


    # 将fuel_type中的not supported和-改为unknown or other
    df['fuel_type'] = df['fuel_type'].apply(lambda x: 'unknown or other (like electricity)' if x in ['not supported', '–'] else x)





# 将model特征下每一个不同类别赋值为这个类别的平均值，假设先验不同model的价值服从同样分布
def assign_model_avg_price(df):
    # 计算每个model的平均价格
    model_avg_price = df.groupby('model')['price'].mean()
    # 将model映射为对应的平均价格
    df['model'] = df['model'].map(model_avg_price)

# 将brand特征下每一个不同类别赋值为这个类别的平均值，假设先验不同brand的价值服从同样分布
def assign_brand_avg_price(df):
    # 计算每个brand的平均价格
    brand_avg_price = df.groupby('brand')['price'].mean()
    # 将brand映射为对应的平均价格
    df['brand'] = df['brand'].map(brand_avg_price)

# 将transmission特征下每一个不同类别赋值为这个类别的平均值，假设先验不同transmission的价值服从同样分布
def assign_transmission_avg_price(df):
    # 计算每个transmission的平均价格
    transmission_avg_price = df.groupby('transmission')['price'].mean()
    # 将transmission映射为对应的平均价格
    df['transmission'] = df['transmission'].map(transmission_avg_price)


# 将fuel_type特征下每一个不同类别赋值为这个类别的平均值，假设先验不同fuel_type的价值服从同样分布

def assign_fuel_type_avg_price(df):
    # 计算每个fuel_type的平均价格
    fuel_type_avg_price = df.groupby('fuel_type')['price'].mean()
    # 将fuel_type映射为对应的平均价格
    df['fuel_type'] = df['fuel_type'].map(fuel_type_avg_price)

# 将ext_col和int_col特征下每一个不同类别赋值为这个类别的平均值，假设先验不同color的价值服从同样分布

def assign_ext_int_avg_price(df):
    # 计算每个ext_col的平均价格
    ext_col_avg_price = df.groupby('ext_col')['price'].mean()
    # 打印每个ext_col的平均价格
    print(ext_col_avg_price)
    # 将ext_col映射为对应的平均价格
    df['ext_col'] = df['ext_col'].map(ext_col_avg_price)



    # 计算每个int_col的平均价格
    int_col_avg_price = df.groupby('int_col')['price'].mean()
    #打印每个int_col的平均价格
    print(int_col_avg_price)
    # 将int_col映射为对应的平均价格
    df['int_col'] = df['int_col'].map(int_col_avg_price)


# one_hot编码，将其转化为数值型数据
def one_hot_encode(df):
    # 选择非数字列
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    # 对非数字列进行 One-Hot 编码，变为0/1
    df1 = pd.get_dummies(df, columns=non_numeric_columns, dtype=int)


    return df1



# 平均值替代+onehot编码
def data_preprocessing2(df):
    drop_id(df)
    drop_duplicates(df)
    update(df)

    add_age_features(df)
    add_engine_features(df)
    fill_missing_values(df)
    preview__(df)
    assign_ext_int_avg_price(df)
    assign_brand_avg_price(df)
    assign_model_avg_price(df)
    assign_transmission_avg_price(df)
    assign_fuel_type_avg_price(df)


    return df

# 直接onehot编码
def data_preprocessing1(df):
    drop_id(df)
    drop_duplicates(df)
    update(df)

    add_age_features(df)
    add_engine_features(df)
    fill_missing_values(df)
    preview__(df)
    assign_ext_int_avg_price(df)
    return df

# 创建一个词典，key是每一个brand加_加这个brand下存在的model，value是出现次数
def create_brand_model_dict(df):
    # 创建一个空字典
    brand_model_dict_default = {}
    # 遍历每一行
    for index, row in df.iterrows():
        # 获取品牌和车型
        brand = row['brand']
        model = row['model']
        # 创建key
        key = f"{brand}_{model}"
        # 如果key已经存在，则将值加1，否则初始化为1
        if key in brand_model_dict_default:
            brand_model_dict_default[key] += 1
        else:
            brand_model_dict_default[key] = 1
    return brand_model_dict_default

# 将所有的brand以及它旗下的model列出dict以便后续取用

def brand_model_dict(df):

    # 创建一个空字典
    brand_model_dict = {}

    brand_model_dict_default = create_brand_model_dict(df)
    # 遍历df的每一行，找到每一个brand和它对应的model



    for index, row in df.iterrows():
        brand = row['brand']
        model = row['model']
        # 如果brand不在字典中，则添加
        if brand_model_dict_default[f"{brand}_{model}"] >= 2:
            if brand not in brand_model_dict:
                brand_model_dict[brand] = []
            # 如果model不在brand对应的列表中，则添加
            if model not in brand_model_dict[brand]:
                brand_model_dict[brand].append(model)
    return brand_model_dict

# 数值等比除以最大值，使其在0-1间
def normalize(df):
    # 储存除的值
    max_values = {}
    # 选择数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    # 对数值列进行归一化处理
    for col in numeric_columns:
        if col not in ['clean_title','is_electric']:
            max_values[col] = df[col].max()
            df[col] = df[col] / df[col].max()
    return max_values



if __name__ == '__main__':
    # 读取数据
    # 获取数据绝对目录
    current_dir = os.getcwd()
    data_dir = r'data\data.csv'
    data_absdir = os.path.join(current_dir, data_dir)
    df1 = pd.read_csv(data_absdir)
    df2 = pd.read_csv(data_absdir)
    df1 = data_preprocessing1(df1)
    df2 = data_preprocessing2(df2)
    brand_model_dict = brand_model_dict(df1)

    print(df1.columns)
    pd.set_option('display.max_columns', None)
    #df1与df2的区别在于df2是将所有可量化的东西全量化了，df1只量化了类似颜色之类的东西
    dic1 = normalize(df1)
    df1 = one_hot_encode(df1)

    dic2 = normalize(df2)
    df2 = one_hot_encode(df2)

    print(df1.shape,df2.shape)
    print(dic1,'\n',dic2)

    # 将数据保存到csv文件中
    df1.to_csv('data/processed_only_one_hot.csv', index=False)
    df2.to_csv('data/processed_with_average.csv', index=False)

    # 将dic1,dic2以及brand_model_dict保存到json文件中
    with open('data/only_one_hot_max_values.json', 'w') as f1:
        json.dump(dic1, f1)
    with open('data/with_average_max_values.json', 'w') as f2:
        json.dump(dic2, f2)
    with open('data/brand_model_dict.json', 'w') as f3:
        json.dump(brand_model_dict, f3)



