import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from enum import IntEnum
import sys, glob, os, math, numpy as np
# 定义一个整数枚举类，用于表示化学反应中的物质角色
class Role(IntEnum):
    REACTANT = 0  # 反应物
    REAGENT = 1   # 试剂
    CATALYST = 2  # 催化剂
    SOLVENT = 3   # 溶剂

# 定义一个数据类，用于存储物质的量的信息
@dataclass
class Amount:
    unit: str  # 单位
    value: float  # 数值

# 定义一个数据类，用于存储物质的信息
@dataclass
class Material:
    name: str  # 名称
    smiles: str  # SMILES 字符串，用于表示分子结构
    amount: Amount  # 物质的量
    role: Role  # 物质在反应中的角色
    def molecule(self):
        return Chem.MolFromSmiles(self.smiles)  # 从SMILES字符串创建分子对象

# 创建一个函数，用于生成分子的Morgan指纹
_mgen = GetMorganGenerator(radius=2)
def fingerprint_of(molecule):
    return list(_mgen.GetFingerprint(molecule))

# 定义一个数据类，用于存储产物的信息
@dataclass
class Product:
    smiles: str  # 产物的SMILES 字符串
    desired: bool  # 是否是期望的产物
    percent: float  # 产物的百分比

# 定义一个函数，用于将输入数据转换为Material对象
def input_as_material(name: str, data: dict) -> Material:
    component = data['components'][0]
    smiles = component['identifiers'][0]['value']
    try:
        amount = Amount('MOLE', component['amount']['moles']['value'])
    except KeyError:
        amount = Amount('LITER', component['amount']['volume']['value'])
    role = getattr(Role, component['reaction_role'])
    return Material(name, smiles, amount, role)

# 定义一个函数，用于将输出数据转换为Product对象
def outcome_as_product(data: dict) -> Product:
    smiles = data['identifiers'][0]['value']
    percent = data['measurements'][0]['percentage']['value']
    desired = data['is_desired_product']
    return Product(smiles, desired, percent)

# 指定包含 JSON 文件的本地文件夹路径
folder_path = r'C:\\Users\\35078\\OneDrive\\文档\\GitHub\\ordjson'

# 创建一个空集合，用于存储所有可能的键
keys = set()

# 检查命令行参数，决定是否使用缓存的数据
if '--cache' not in sys.argv:
    data_list = []

    for filename in glob.iglob(os.path.join(folder_path, '*.json')):
        data = json.load(open(filename))
        inputs = [input_as_material(name, inp) for name, inp in data['inputs'].items()]
        d = {
            'temperature': data['conditions']['temperature']['setpoint']['value'],
            'Solvent': 0
        }
        data_list.append(d)
        for material in inputs:
            d[material.name] = material.amount.value
            if material.role != Role.REACTANT:
                molecule = material.molecule()
                for name, desc in Descriptors.descList:
                    d[f'{material.name}::{name}'] = desc(molecule)
                for index, fg in enumerate(fingerprint_of(molecule)):
                    d[f'{material.name}::fingerprint_{index}'] = fg
        product = outcome_as_product(data['outcomes'][0]['products'][0])
        d['product'] = product.percent
        keys |= d.keys()
    
    # 计算平均温度并填补缺失的温度数据
    avg_temp = np.average([data['temperature'] for data in data_list if isinstance(data['temperature'], (int, float))])
    for row in data_list:
        if not isinstance(row['temperature'], (int, float)):
            row['temperature'] = avg_temp
        for key in keys:
            if key not in row or math.isnan(row[key]):
                row[key] = 0

    # 将数据转换为pandas DataFrame
    df = pd.DataFrame(data_list)

    # 尝试将数据保存为CSV文件
    while True:
        try:
            df.to_csv(os.path.join(folder_path, 'data_new.csv'))
        except PermissionError:
            input('Cannot save file. Press Enter to retry.')
            continue
        else:
            break

else:
    # 如果使用缓存，则直接从CSV文件读取数据
    df = pd.read_csv(os.path.join(folder_path, 'data_new.csv'))

# 分离特征和目标变量
X = df.drop(columns=['product'])
y = df['product']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 初始化并训练随机森林回归模型
model = RandomForestRegressor(n_estimators=1000)

model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

sys.exit()