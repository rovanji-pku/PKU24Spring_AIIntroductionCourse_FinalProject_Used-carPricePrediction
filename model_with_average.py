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
    data_dir_train = r'data\train_with_average.csv'
    data_dir_test = r'data\test_with_average.csv'
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
    model_dir = r'model_1'
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
均方误差 (MSE): 4702939958.582307
均方根误差 (RMSE): 68577.98450364597
平均绝对误差 (MAE): 21428.674757106146
R^2: 0.15432025224924195
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\LinearRegression.pkl
当前回归模型是: Ridge
均方误差 (MSE): 4702936280.181138
均方根误差 (RMSE): 68577.95768452964
平均绝对误差 (MAE): 21428.1429659583
R^2: 0.15432091369705958
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\Ridge.pkl
当前回归模型是: Lasso
均方误差 (MSE): 5561139921.156089
均方根误差 (RMSE): 74573.05090417106
平均绝对误差 (MAE): 29086.84720393411
R^2: -7.329771751773961e-07
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\Lasso.pkl
当前回归模型是: ElasticNet
均方误差 (MSE): 5561139921.156089
均方根误差 (RMSE): 74573.05090417106
平均绝对误差 (MAE): 29086.84720393411
R^2: -7.329771751773961e-07
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\ElasticNet.pkl
当前回归模型是: BayesianRidge
均方误差 (MSE): 4702936275.291833
均方根误差 (RMSE): 68577.95764888186
平均绝对误差 (MAE): 21427.559200049065
R^2: 0.1543209145762514
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\BayesianRidge.pkl
当前回归模型是: DecisionTreeRegressor
均方误差 (MSE): 12547038765.419588
均方根误差 (RMSE): 112013.5650955704
平均绝对误差 (MAE): 28859.35416279913
R^2: -1.2562007322240234
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\DecisionTreeRegressor.pkl
当前回归模型是: GradientBoostingRegressor
均方误差 (MSE): 4764833294.209925
均方根误差 (RMSE): 69027.77190529856
平均绝对误差 (MAE): 19856.914136429867
R^2: 0.14319063100763985
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\GradientBoostingRegressor.pkl
当前回归模型是: BaggingRegressor
均方误差 (MSE): 5807918604.612553
均方根误差 (RMSE): 76209.70151242263
平均绝对误差 (MAE): 22843.624991553508
R^2: -0.044376322845143035
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\BaggingRegressor.pkl
当前回归模型是: AdaBoostRegressor
均方误差 (MSE): 4774301617.14641
均方根误差 (RMSE): 69096.32129966407
平均绝对误差 (MAE): 21474.930531092752
R^2: 0.14148804304711582
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\AdaBoostRegressor.pkl
当前回归模型是: MLPRegressor
均方误差 (MSE): 4675192475.636046
均方根误差 (RMSE): 68375.37916264923
平均绝对误差 (MAE): 20885.546506545747
R^2: 0.1593097874305045
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\MLPRegressor.pkl
当前回归模型是: RandomForestRegressor
均方误差 (MSE): 5665358060.390603
均方根误差 (RMSE): 75268.57286006294
平均绝对误差 (MAE): 22016.09932891813
R^2: -0.018741174163982244
模型已保存到： C:\Users\楼文颉\Desktop\通识课\人工智能与计算思维\project_file\model_1\RandomForestRegressor.pkl
'''











