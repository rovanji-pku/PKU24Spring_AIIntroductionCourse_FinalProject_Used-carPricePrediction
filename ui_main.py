import tkinter as tk
from tkinter import ttk
import os
import json
import pandas as pd
import joblib

column_lis = ['brand', 'model', 'model_year','milage', 'Horsepower', 'Engine_Size','transmission',
              'fuel_type', 'ext_col', 'int_col', 'accident', 'clean_title', 'is_electric']
ext_col_dict = {
    "Agate Black Metallic": 59775.414710,
    "Alpine White": 76781.226481,
    "Beige": 25639.143514,
    "Black": 42417.511447,
    "Black Clearcoat": 36192.640411,
    "Blue": 40658.658811,
    "Bright White Clearcoat": 47630.185430,
    "Brown": 33947.655766,
    "Diamond Black": 61810.690577,
    "Ebony Twilight Metallic": 40614.159170,
    "Gold": 23090.873501,
    "Granite Crystal Clearcoat Metallic": 48302.602740,
    "Gray": 47908.967106,
    "Green": 53064.767976,
    "Midnight Black Metallic": 47517.517150,
    "Mythos Black Metallic": 56145.666667,
    "Obsidian Black Metallic": 81447.494881,
    "Orange": 54180.219491,
    "Oxford White": 85245.826797,
    "Purple": 31671.734463,
    "Red": 39977.708413,
    "Santorini Black Metallic": 96115.637771,
    "Silver": 31878.405296,
    "Silver Ice Metallic": 58935.493421,
    "Summit White": 53071.836576,
    "White": 40803.542622,
    "Yellow": 43366.336401,
    "other": 67485.969093
}
int_col_dict = {
    "Beige": 29953.997428,
    "Black": 45524.117958,
    "Blue": 60456.454913,
    "Brown": 47406.989501,
    "Ebony": 49470.199127,
    "Global Black": 48156.652866,
    "Gray": 28182.418129,
    "Jet Black": 57881.426188,
    "Orange": 60193.241416,
    "Red": 59286.907872,
    "White": 57884.472907,
    "other": 63709.392095
}
milage_with_age_dict = {
    1: 14634.018519,
    2: 9773.009123,
    3: 17824.175440,
    4: 29346.041378,
    5: 34423.869006,
    6: 46342.528522,
    7: 51728.104423,
    8: 67940.690480,
    9: 76157.720356,
    10: 82115.020546,
    11: 87728.537017,
    12: 92865.336816,
    13: 103217.245119,
    14: 113414.643735,
    15: 110100.663991,
    16: 108313.011046,
    17: 115388.304112,
    18: 118515.705906,
    19: 118365.962687,
    20: 121495.681687,
    21: 131982.563813,
    22: 118032.225079,
    23: 118470.721529,
    24: 110134.525424,
    25: 132044.431900,
    26: 134483.140794,
    27: 119915.676385,
    28: 128988.118590,
    29: 111848.449735,
    30: 108466.990196,
    31: 110032.728261,
    32: 100521.728571,
    33: 93404.785714,
    51: 67654.750000
}
Mileage_per_Year_with_age_dict = vehicle_age_dict = {
    1: 14634.018519,
    2: 4886.504562,
    3: 5941.391813,
    4: 7336.510345,
    5: 6884.773801,
    6: 7723.754754,
    7: 7389.729203,
    8: 8492.586310,
    9: 8461.968928,
    10: 8211.502055,
    11: 7975.321547,
    12: 7738.778068,
    13: 7939.788086,
    14: 8101.045981,
    15: 7340.044266,
    16: 6769.563190,
    17: 6787.547301,
    18: 6584.205884,
    19: 6229.787510,
    20: 6074.784084,
    21: 6284.883991,
    22: 5365.101140,
    23: 5150.900936,
    24: 4588.938559,
    25: 5281.777276,
    26: 5172.428492,
    27: 4441.321348,
    28: 4606.718521,
    29: 3856.843094,
    30: 3615.566340,
    31: 3549.442847,
    32: 3141.304018,
    33: 2830.448052,
    51: 1326.563725
}

# 从data\brand_model_dict.json读取dict
current_dir = os.getcwd()
dict_dir = r'data\brand_model_dict.json'
dict_absdir = os.path.join(current_dir, dict_dir)
with open(dict_absdir, 'r', encoding='utf-8') as f:
    options_dict = json.load(f)
only_one_hot_dir = os.path.join(current_dir, r'data\test_only_one_hot.csv')
one_hot_columns_lis = pd.read_csv(only_one_hot_dir).columns.tolist()
# 将price项删除
one_hot_columns_lis.remove('price')

def validate_integer(value):
    return value.isdigit()
def validate_float(value):
    return value.replace(".", "", 1).isdigit()


def update_second_input():
    # 根据第一个输入项的选择更新第二个输入项
    selected_option = var1.get()
    second_options = options_dict.get(selected_option, [])
    var2.set('')  # 清空第二个输入项的选择
    for widget in frame2.winfo_children():
        widget.destroy()  # 清除旧的选项
    for option in second_options:
        ttk.Radiobutton(frame2, text=option, variable=var2, value=option).pack(anchor="w")


root = tk.Tk()
root.title('used-car price prediction base on american car dataset')


# 做废掉的部分
'''
frame1 = ttk.LabelFrame(root, text='brand')
frame1.pack(padx=10, pady=5, fill='x')
var1 = tk.StringVar(value='')
for option in options_dict.keys():
    ttk.Radiobutton(frame1, text=option, variable=var1, value=option, command=update_second_input).pack(anchor="w")

frame2 = ttk.LabelFrame(root, text="model")
frame2.pack(padx=10, pady=5, fill="x")
var2 = tk.StringVar(value="")
update_second_input()  # 初始化第二个输入项


# 注册验证函数
vcmd1 = (root.register(validate_integer), '%P')
vcmd2 = (root.register(validate_float), '%P')
#输入model_year的年份，确保其是整数
frame3 = ttk.LabelFrame(root, text='model_year')
frame3.pack(padx=10, pady=5, fill='x')
var3 = tk.StringVar(value='')
model_year_entry = ttk.Entry(frame3, textvariable=var3, validate='key', validatecommand=vcmd1)
model_year_entry.pack(padx=10, pady=5, fill='x')



# 输入milage
frame4 = ttk.LabelFrame(root, text='milage')
frame4.pack(padx=10, pady=5, fill='x')
var4 = tk.StringVar(value='')
milage_entry = ttk.Entry(frame4, textvariable=var4,validate = 'key', validatecommand=vcmd1)
milage_entry.pack(padx=10, pady=5, fill='x')

# 输入Horsepower
frame5 = ttk.LabelFrame(root, text='Horsepower')
frame5.pack(padx=10, pady=5, fill='x')
var5 = tk.StringVar(value='')
horsepower_entry = ttk.Entry(frame5, textvariable=var5, validate='key', validatecommand=vcmd1)
horsepower_entry.pack(padx=10, pady=5, fill='x')

# 输入Engine_Size
frame6 = ttk.LabelFrame(root, text='Engine_Size')
frame6.pack(padx=10, pady=5, fill='x')
var6 = tk.StringVar(value='')
engine_size_entry = ttk.Entry(frame6, textvariable=var6, validate='key', validatecommand=vcmd2)
engine_size_entry.pack(padx=10, pady=5, fill='x')

# 选择transmission
frame7 = ttk.LabelFrame(root, text='transmission')
frame7.pack(padx=10, pady=5, fill='x')
var7 = tk.StringVar(value='')
transmission_options = [
    "A/T", "8-Speed A/T", "Transmission w/Dual Shift Mode", "6-Speed A/T", "6-Speed M/T",
    "7-Speed A/T", "Automatic", "8-Speed Automatic", "10-Speed A/T", "9-Speed A/T",
    "5-Speed A/T", "10-Speed Automatic", "6-Speed Automatic", "4-Speed A/T", "other",
    "5-Speed M/T", "9-Speed Automatic", "CVT Transmission", "1-Speed A/T", "M/T",
    "7-Speed Automatic with Auto-Shift", "Automatic CVT", "8-Speed Automatic with Auto-Shift"
]

for option in transmission_options:
    ttk.Radiobutton(frame7, text=option, variable=var7, value=option).pack(anchor="w")

# 选择fuel_type
frame8 = ttk.LabelFrame(root, text='fuel_type')
frame8.pack(padx=10, pady=5, fill='x')
var8 = tk.StringVar(value='')
fuel_type_options = ["Gasoline", "Hybrid", "unknown or other (like electricity)", "E85 Flex Fuel", "Diesel", "Plug-In Hybrid"]
for option in fuel_type_options:
    ttk.Radiobutton(frame8, text=option, variable=var8, value=option).pack(anchor="w")

# 选择ext_col
frame9 = ttk.LabelFrame(root, text='ext_col')
frame9.pack(padx=10, pady=5, fill='x')
var9 = tk.StringVar(value='')
ext_col_options = [
    "Agate Black Metallic", "Alpine White", "Beige", "Black", "Black Clearcoat", "Blue",
    "Bright White Clearcoat", "Brown", "Diamond Black", "Ebony Twilight Metallic", "Gold",
    "Granite Crystal Clearcoat Metallic", "Gray", "Green", "Midnight Black Metallic",
    "Mythos Black Metallic", "Obsidian Black Metallic", "Orange", "Oxford White", "Purple",
    "Red", "Santorini Black Metallic", "Silver", "Silver Ice Metallic", "Summit White",
    "White", "Yellow", "other"
]
for option in ext_col_options:
    ttk.Radiobutton(frame9, text=option, variable=var9, value=option).pack(anchor="w")

# 选择int_col
frame10 = ttk.LabelFrame(root, text='int_col')
frame10.pack(padx=10, pady=5, fill='x')
var10 = tk.StringVar(value='')
int_col_options =[
    "Beige", "Black", "Blue", "Brown", "Ebony", "Global Black", "Gray",
    "Jet Black", "Orange", "Red", "White", "other"
]
for option in int_col_options:
    ttk.Radiobutton(frame10, text=option, variable=var10, value=option).pack(anchor="w")

# 选择accident
frame11 = ttk.LabelFrame(root, text='accident')
frame11.pack(padx=10, pady=5, fill='x')
var11 = tk.StringVar(value='')
accident_options = [
    "None reported",
    "At least 1 accident or damage reported",
    "unknown or unwilling to dispose"
]
for option in accident_options:
    ttk.Radiobutton(frame11, text=option, variable=var11, value=option).pack(anchor="w")

# 选择clean_title
frame12 = ttk.LabelFrame(root, text='clean_title')
frame12.pack(padx=10, pady=5, fill='x')
var12 = tk.StringVar(value='')
clean_title_options = [
    'Yes','No or unknown'
]
for option in clean_title_options:
    ttk.Radiobutton(frame12, text=option, variable=var12, value=option).pack(anchor="w")

# 选择is_electric
frame13 = ttk.LabelFrame(root, text='is_electric')
frame13.pack(padx=10, pady=5, fill='x')
var13 = tk.StringVar(value='')
is_electric_options = [
    'Yes','No or unknown'
]
for option in is_electric_options:
    ttk.Radiobutton(frame13, text=option, variable=var13, value=option).pack(anchor="w")
'''
# 注册验证函数
vcmd1 = (root.register(validate_integer), '%P')
vcmd2 = (root.register(validate_float), '%P')

def sort_options(options,arg):
    # 将 "other" 移到最后
    options = sorted([opt for opt in options if opt != str(arg)]) + [str(arg)]
    return options
def sort_options_1(options):
    # 直接排序
    options = sorted(options)
    return options
# 定义选项
transmission_options = sort_options([
    "A/T", "8-Speed A/T", "Transmission w/Dual Shift Mode", "6-Speed A/T", "6-Speed M/T",
    "7-Speed A/T", "Automatic", "8-Speed Automatic", "10-Speed A/T", "9-Speed A/T",
    "5-Speed A/T", "10-Speed Automatic", "6-Speed Automatic", "4-Speed A/T", "other",
    "5-Speed M/T", "9-Speed Automatic", "CVT Transmission", "1-Speed A/T", "M/T",
    "7-Speed Automatic with Auto-Shift", "Automatic CVT", "8-Speed Automatic with Auto-Shift"
],"other")
fuel_type_options = sort_options(["Gasoline", "Hybrid", "unknown or other (like electricity)", "E85 Flex Fuel", "Diesel", "Plug-In Hybrid"],"unknown or other (like electricity)")
ext_col_options = sort_options([
    "Agate Black Metallic", "Alpine White", "Beige", "Black", "Black Clearcoat", "Blue",
    "Bright White Clearcoat", "Brown", "Diamond Black", "Ebony Twilight Metallic", "Gold",
    "Granite Crystal Clearcoat Metallic", "Gray", "Green", "Midnight Black Metallic",
    "Mythos Black Metallic", "Obsidian Black Metallic", "Orange", "Oxford White", "Purple",
    "Red", "Santorini Black Metallic", "Silver", "Silver Ice Metallic", "Summit White",
    "White", "Yellow", "other"
],"other")
int_col_options = sort_options([
    "Beige", "Black", "Blue", "Brown", "Ebony", "Global Black", "Gray",
    "Jet Black", "Orange", "Red", "White", "other"
],"other")
accident_options = [
    "None reported",
    "At least 1 accident or damage reported",
    "unknown or unwilling to dispose"
]
clean_title_options = ['Yes', 'No or unknown']
is_electric_options = ['Yes', 'No or unknown']

# 创建输入框和下拉框
def create_combobox(frame, label_text, options):
    label = ttk.Label(frame, text=label_text)
    label.pack(anchor="w", padx=10, pady=2)
    combobox = ttk.Combobox(frame, values=options, state="readonly")
    combobox.pack(fill="x", padx=10, pady=2)

    def adjust_combobox_width(combobox, options):
        max_width = max(len(option) for option in options)  # 获取最长选项的长度
        combobox.config(width=max_width)
    if label_text != "Model":
        adjust_combobox_width(combobox, options)

    return combobox

frame = ttk.Frame(root)
frame.pack(padx=20, pady=10, fill="both", expand=True)

brand_combobox = create_combobox(frame, "Brand", sort_options_1(options_dict.keys()))
model_combobox = create_combobox(frame, "Model", [])
brand_combobox.bind("<<ComboboxSelected>>", lambda x: model_combobox.config(values=sort_options(options_dict.get(brand_combobox.get(), []),"other")))

model_year_entry = create_combobox(frame, "Model Year", [str(year) for year in range(2024,1979,-1)])
milage_label = ttk.Label(frame, text="Milage")
milage_label.pack(anchor="w", padx=10, pady=2)
milage_entry = ttk.Entry(frame, validate='key', validatecommand=vcmd1)
milage_entry.pack(fill="x", padx=10, pady=2)
horsepower_label = ttk.Label(frame, text="Horsepower")
horsepower_label.pack(anchor="w", padx=10, pady=2)
horsepower_entry = ttk.Entry(frame, validate='key', validatecommand=vcmd1)
horsepower_entry.pack(fill="x", padx=10, pady=2)
engine_size_label = ttk.Label(frame, text="Engine Size")
engine_size_label.pack(anchor="w", padx=10, pady=2)
engine_size_entry = ttk.Entry(frame, validate='key', validatecommand=vcmd2)
engine_size_entry.pack(fill="x", padx=10, pady=2)
transmission_combobox = create_combobox(frame, "Transmission", transmission_options)
fuel_type_combobox = create_combobox(frame, "Fuel Type", fuel_type_options)
ext_col_combobox = create_combobox(frame, "Exterior Color", ext_col_options)
int_col_combobox = create_combobox(frame, "Interior Color", int_col_options)
accident_combobox = create_combobox(frame, "Accident", accident_options)
clean_title_combobox = create_combobox(frame, "Clean Title", clean_title_options)
is_electric_combobox = create_combobox(frame, "Is Electric", is_electric_options)
# 将窗口最大化
root.state('zoomed')








# 运用模型计算输出
def calculate_output():
    # 将输入的值转换为适当的类型
    brand = brand_combobox.get()
    model = model_combobox.get()
    Vehicle_Age = 2025 - int(model_year_entry.get())
    milage = float(milage_entry.get())
    Horsepower = float(horsepower_entry.get())
    Engine_size = float(engine_size_entry.get())
    Power_to_Weight_Ratio = Horsepower / Engine_size
    transmission = transmission_combobox.get()
    fuel_type = fuel_type_combobox.get()
    ext_col = ext_col_combobox.get()
    int_col = int_col_combobox.get()
    accident = accident_combobox.get()
    clean_title = clean_title_combobox.get()
    is_electric = is_electric_combobox.get()
    # 处理数据
    clean_title = 1 if clean_title == 'Yes' else 0
    is_electric = 1 if is_electric == 'Yes' else 0
    ext_col = ext_col_dict.get(ext_col, 67485.969093)
    int_col = int_col_dict.get(int_col, 63709.392095)
    Mileage_per_Year = milage / Vehicle_Age
    milage_with_age = milage_with_age_dict.get(Vehicle_Age) if Vehicle_Age in milage_with_age_dict.keys() else 93404.785714-(Vehicle_Age-33)*1430.557540
    Mileage_per_Year_with_age = Mileage_per_Year_with_age_dict.get(Vehicle_Age) if Vehicle_Age in Mileage_per_Year_with_age_dict.keys() else 2830.448052-(Vehicle_Age-33)*83.549129
    milage /= 405000
    Horsepower/= 1020
    Engine_size /= 8.4
    Power_to_Weight_Ratio /= 261.53846153846155
    ext_col /= 96115.63777089784
    int_col /= 63709.39209467822
    Mileage_per_Year /= 117500.0
    milage_with_age /= 134483.14079422381
    Mileage_per_Year_with_age /= 14634.018518518518
    Vehicle_Age /= 51.0
    # 将数据转换为DataFrame
    data = {
        'brand': [brand],
        'model': [model],
        'milage': [milage],
        'fuel_type': [fuel_type],
        'transmission': [transmission],
        'ext_col': [ext_col],
        'int_col': [int_col],
        'accident': [accident],
        'clean_title': [clean_title],
        'Vehicle_Age': [Vehicle_Age],
        'Mileage_per_Year': [Mileage_per_Year],
        'milage_with_age': [milage_with_age],
        'Mileage_per_Year_with_age': [Mileage_per_Year_with_age],
        'Horsepower': [Horsepower],
        'Engine_Size': [Engine_size],
        'Power_to_Weight_Ratio': [Power_to_Weight_Ratio],
        'is_electric': [is_electric]
    }
    df = pd.DataFrame(data)
    # 将数据转换为one-hot编码
    df = pd.get_dummies(df, columns=['brand', 'model', 'transmission', 'fuel_type', 'accident'])
    # 构建一行的dataframe,columns使用one_hot_columns_lis，每个值都为0
    one_hot_df = pd.DataFrame(0, index=[0], columns=one_hot_columns_lis)
    # 说明df的每个column都是one_hot_columns_lis的子集
    def is_subset(df, one_hot_columns_lis):
        return all(col in one_hot_columns_lis for col in df.columns)

    print(df.columns)
    if is_subset(df, one_hot_columns_lis):
        print("df的每个column都是one_hot_columns_lis的子集")
        for col in df.columns:
            if col in one_hot_df.columns:
                one_hot_df[col] = df[col]
    # 否则找出不在one_hot_columns_lis中的df的column
    else:
        for col in df.columns:
            if col not in one_hot_df.columns:
                print(f"{col}不在one_hot_columns_lis中")



    # 将one_hot_df喂给模型
    # 读取模型
    model_dir = ['model/only_one_hot_best_model.pkl',
                 'model/AdaBoostRegressor.pkl',
                 'model/BaggingRegressor.pkl',
                 'model/BayesianRidge.pkl',
                 'model/DecisionTreeRegressor.pkl',
                 'model/ElasticNet.pkl',
                 'model/GradientBoostingRegressor.pkl',
                 'model/Lasso.pkl',
                 'model/LinearRegression.pkl',
                 'model/MLPRegressor.pkl',
                 'model/RandomForestRegressor.pkl',
                 'model/Ridge.pkl']

    model_absdir = os.path.join(current_dir, model_dir[0])# 这个数字代表了使用几号模型--开发者选项
    model = joblib.load(model_absdir)
    # 预测
    prediction = float(model.predict(one_hot_df))*2954083
    # 显示输出结果
    output_label.config(text=f"result is {prediction:.2f} $")



# 计算按钮
calculate_button = ttk.Button(root, text="calculate", command=calculate_output)
calculate_button.pack(pady=10)

# 输出
output_label = ttk.Label(root, text="", font=("Arial", 16))
output_label.pack(pady=10)


root.mainloop()







