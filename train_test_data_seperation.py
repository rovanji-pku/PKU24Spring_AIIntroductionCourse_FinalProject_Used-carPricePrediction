import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split



# 将数据分为训练集和测试集
def train_test_split_data(df, test_size=0.2, random_state=42):
    # 将数据分为训练集和测试集
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

if __name__ == '__main__':
    current_dir = os.getcwd()
    data_dir_1 = r'data\processed_only_one_hot.csv'
    data_dir_2 = r'data\processed_with_average.csv'
    data_absdir_1 = os.path.join(current_dir, data_dir_1)
    data_absdir_2 = os.path.join(current_dir, data_dir_2)
    df1 = pd.read_csv(data_absdir_1)
    df2 = pd.read_csv(data_absdir_2)
    train_df_1, test_df_1 = train_test_split_data(df1)
    train_df_2, test_df_2 = train_test_split_data(df2)
    # 将数据分别保存到csv文件中
    train_df_1.to_csv('data/train_only_one_hot.csv', index=False)
    test_df_1.to_csv('data/test_only_one_hot.csv', index=False)
    train_df_2.to_csv('data/train_with_average.csv', index=False)
    test_df_2.to_csv('data/test_with_average.csv', index=False)

