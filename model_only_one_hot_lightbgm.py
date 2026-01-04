import optuna
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd


# 定义Optuna的目标函数
def objective(trial):
    # 使用Optuna选择超参数
    params = {
        'objective': 'regression',  # 回归任务
        'boosting_type': 'gbdt',  # 梯度提升决策树
        'num_leaves': trial.suggest_int('num_leaves', 50, 200),  # 树的最大叶子数
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),  # 学习率，取对数均匀分布
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # 树的数量
        'max_depth': trial.suggest_int('max_depth', 5, 30),  # 树的最大深度
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # 数据采样率
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # 特征采样率
        # 添加L1和L2正则化参数
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),  # L1正则化强度
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0)  # L2正则化强度
    }

    # 创建模型，设置 verbose 为 False 以避免输出日志
    model = lgb.LGBMRegressor(**params, verbose=-1)

    # 训练模型，使用 callbacks 参数进行早期停止
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # 进行预测
    y_pred = model.predict(X_test)

    # 计算 RMSE（均方根误差）
    rmse = np.sqrt(mean_squared_error(y_test*2954083, y_pred*2954083))

    # 记录每次试验的 RMSE
    rmse_list.append(rmse)

    return rmse  # Optuna 会根据最小化RMSE来寻找最佳超参数




if __name__ == "__main__":
    current_dir = os.getcwd()
    data_dir_train = r'data\train_only_one_hot.csv'
    data_dir_test = r'data\test_only_one_hot.csv'
    data_absdir_train = os.path.join(current_dir, data_dir_train)
    data_absdir_test = os.path.join(current_dir, data_dir_test)
    df1 = pd.read_csv(data_absdir_train)
    df2 = pd.read_csv(data_absdir_test)

    # 划分数据集
    y_train, X_train, = df1['price'], df1.drop(columns=['price'])
    y_test, X_test = df2['price'], df2.drop(columns=['price'])
    '''
    # 标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 计算训练集的均值和标准差，并应用于训练集
    X_test_scaled = scaler.transform(X_test)  # 使用训练集的均值和标准差对验证集进行转换
    '''
    # 记录每次试验的 RMSE
    rmse_list = []

    # 创建Optuna的Study对象
    study = optuna.create_study(direction='minimize')  # 最小化RMSE

    # 开始超参数优化
    study.optimize(objective, n_trials=200)  # 尝试最多200次

    # 输出最佳超参数和对应的RMSE值
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best RMSE: {study.best_value}")

    # 使用最佳超参数训练最终模型
    best_params = study.best_trial.params
    final_model = lgb.LGBMRegressor(**best_params, verbose=-1)
    final_model.fit(X_train, y_train)  # 在整个训练集上训练
    y_test = y_test*2954083
    # 在验证集上进行预测并计算RMSE
    y_pred_final = final_model.predict(X_test)*2954083
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

    print(f"Final RMSE on test set: {final_rmse}")

    # 绘制每次试验的 RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rmse_list) + 1), rmse_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Trial Number')
    plt.ylabel('RMSE')
    plt.title('RMSE Progression During Optuna Optimization')
    plt.grid(True)
    plt.show()

    # 存储模型参数
    import joblib

    model_dir = r'model'
    model_absdir = os.path.join(current_dir, model_dir)
    if not os.path.exists(model_absdir):
        os.makedirs(model_absdir)
    model_path = os.path.join(model_absdir, 'only_one_hot_best_model.pkl')
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")



    #####注意之后做ui的部分要考虑normalize之后才能扔进model
    #####之前数据处理 于是把正则化部件删除了
