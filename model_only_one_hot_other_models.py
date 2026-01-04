import numpy as np
import pandas as pd
import os
import sklearn as skl
import joblib
'''import cuml
from cuml.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from cuml.tree import DecisionTreeRegressor
from cuml.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor,AdaBoostRegressor
from cuml.neural_network import MLPRegressor
from cuml.metrics import mean_squared_error, accuracy_score,r2_score,mean_absolute_error'''
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score,r2_score,mean_absolute_error


def try_different_method(reg):
    reg.fit(X_train, y_train)
    #将test集传入模型进行预测
    y_pred = reg.predict(X_test)*2954083
    # 计算误差
    y_test_1 = y_test*2954083
    mse = mean_squared_error(y_test_1, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_1, y_pred)
    r2 = r2_score(y_test_1, y_pred)
    # 打印结果
    print("均方误差 (MSE):", mse)
    print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)
    print("R^2:", r2)
    return reg



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

    # 不同分类器
    regs = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'BayesianRidge': BayesianRidge(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'BaggingRegressor': BaggingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'MLPRegressor': MLPRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
    }
    model_dir = r'model'
    model_absdir = os.path.join(current_dir, model_dir)
    if not os.path.exists(model_absdir):
        os.makedirs(model_absdir)
    # 不同模型训练
    for reg_key in regs.keys() :

        print("当前回归模型是:", reg_key)
        reg = regs[reg_key]
        save_reg = try_different_method(reg)
        # 保存模型

        dir = os.path.join(model_absdir, reg_key + '.pkl')
        joblib.dump(save_reg, dir)
        print("模型已保存到：", dir)
'''
当前回归模型是: LinearRegression
均方误差 (MSE): 4748388957.250926
均方根误差 (RMSE): 68908.55503673638
平均绝对误差 (MAE): 21075.26868184324
R^2: 0.1461476414848918
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\LinearRegression.pkl
当前回归模型是: Ridge
均方误差 (MSE): 4741674473.838217
均方根误差 (RMSE): 68859.81755594634
平均绝对误差 (MAE): 21040.02637254249
R^2: 0.14735503572950115
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\Ridge.pkl
当前回归模型是: Lasso
均方误差 (MSE): 5561139921.156089
均方根误差 (RMSE): 74573.05090417106
平均绝对误差 (MAE): 29086.84720393411
R^2: -7.329771751773961e-07
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\Lasso.pkl
当前回归模型是: ElasticNet
均方误差 (MSE): 5561139921.156089
均方根误差 (RMSE): 74573.05090417106
平均绝对误差 (MAE): 29086.84720393411
R^2: -7.329771751773961e-07
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\ElasticNet.pkl
当前回归模型是: BayesianRidge
均方误差 (MSE): 4720380657.053601
均方根误差 (RMSE): 68705.0264322313
平均绝对误差 (MAE): 20879.91351455124
R^2: 0.15118407666254619
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\BayesianRidge.pkl
当前回归模型是: DecisionTreeRegressor
均方误差 (MSE): 11419958127.959862
均方根误差 (RMSE): 106864.20414694464
平均绝对误差 (MAE): 27366.030714898206
R^2: -1.0535297907329846
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\DecisionTreeRegressor.pkl
当前回归模型是: GradientBoostingRegressor
均方误差 (MSE): 4882405424.893624
均方根误差 (RMSE): 69874.2114438054
平均绝对误差 (MAE): 20358.57928457103
R^2: 0.12204888335728659
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\GradientBoostingRegressor.pkl
当前回归模型是: BaggingRegressor
均方误差 (MSE): 6616826631.09239
均方根误差 (RMSE): 81343.87887906742
平均绝对误差 (MAE): 22562.03016587485
R^2: -0.189833662681828
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\BaggingRegressor.pkl
当前回归模型是: AdaBoostRegressor
均方误差 (MSE): 15465404225.210829
均方根误差 (RMSE): 124359.97839019926
平均绝对误差 (MAE): 70456.38097869996
R^2: -1.780979399954402
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\AdaBoostRegressor.pkl
当前回归模型是: MLPRegressor
均方误差 (MSE): 4746774212.967691
均方根误差 (RMSE): 68896.83746709781
平均绝对误差 (MAE): 21701.381825141598
R^2: 0.14643800380083738
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\MLPRegressor.pkl
当前回归模型是: RandomForestRegressor
均方误差 (MSE): 6017262508.929007
均方根误差 (RMSE): 77571.01590754763
平均绝对误差 (MAE): 21466.23416585897
R^2: -0.08202041393595616
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model\RandomForestRegressor.pkl
'''