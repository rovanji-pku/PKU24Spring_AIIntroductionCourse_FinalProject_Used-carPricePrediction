import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt


def preview(df):
    # 预览数据
    print(df.isnull().sum())
    pd.set_option('display.max_columns', None)
    print(df.head())
    print(df.describe())

    # 统计不同值
    categorical_columns = df.select_dtypes(include = ['object','int']).columns
    unique_values = {col: df[col].nunique() for col in categorical_columns}
    for col, unique_count in unique_values.items():
        print(f"{col}: {unique_count} unique values")


    # 统计车型数据预览
    plt.figure(figsize=(12, 6))
    sns.barplot(x='brand', y='price', data=df[:10000], errorbar=None)
    plt.title('Average Price by Car Brand')
    plt.xlabel('Brand')
    plt.ylabel('Average Price')
    plt.xticks(rotation=90)
    plt.show()

    #变速器影响
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='price', y='transmission', data=df[:1000], orient='h')
    plt.title('Box Plot of Price by Transmission Type')
    plt.xlabel('Transmission')
    plt.ylabel('Price')
    plt.xticks(rotation=90)
    plt.show()

    # 统计事故影响
    plt.figure(figsize=(10, 6))
    sns.barplot(x='accident', y='price', data=df, errorbar=None)
    plt.title('Average Price by Accident History')
    plt.xlabel('Accident History')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.show()


    #na值
    missing_values = df.isnull().mean() * 100
    missing_values = missing_values[missing_values > 0]
    missing_values = missing_values.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Percentage of Missing Values')
    plt.title('Missing Values Distribution in df_train')
    plt.show()
    '''
    # 相关度分析
    from dython.nominal import associations

    associations_df = associations(df, nominal_columns='all', plot=False, cramers_v_bias_correction=False)
    corr_matrix = associations_df['corr']
    plt.figure(figsize=(20, 8))
    plt.gcf().set_facecolor('#FFFDD0')
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix including Categorical Features')
    plt.show()
    '''
def preview__(df):
    print(df.isnull().sum())
    pd.set_option('display.max_columns', None)



    # 统计不同值
    categorical_columns = df.select_dtypes(include=['object', 'int']).columns
    unique_values = {col: df[col].nunique() for col in categorical_columns}
    for col, unique_count in unique_values.items():
        print(f"{col}: {unique_count} unique values")

    # 查看fuel_type下的不同值分别是多少
    fuel_type_counts = df['fuel_type'].value_counts()
    print(fuel_type_counts)
    # 查看transmission下的不同值分别是多少
    transmission_counts = df['transmission'].value_counts()
    print(transmission_counts)
    # 查看accident下的不同值分别是多少
    accident_counts = df['accident'].value_counts()
    print(accident_counts)




if __name__ == '__main__':
    # 读取数据
    # 获取数据绝对目录
    current_dir = os.getcwd()
    data_dir = r'data\data.csv'
    data_absdir = os.path.join(current_dir, data_dir)
    df = pd.read_csv(data_absdir)

    preview__(df)


