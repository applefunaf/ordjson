import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 指定包含 JSON 文件的本地文件夹路径
folder_path = 'C:\\Users\\35078\\OneDrive\\桌面\\大二上\\暑期程设\\课堂\\大作业\\00'

# 初始化一个空的列表来存储数据
data_list = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # 提取 SMILES、reagent、catalyst 和 yield 信息
            smiles_list = []
            reagents = []
            catalysts = []
            yield_value = None
            
            for input_type, input_data in data['inputs'].items():
                for component in input_data['components']:
                    for identifier in component['identifiers']:
                        if identifier['type'] == 'SMILES':
                            smiles_list.append(identifier['value'])
                    if input_type == 'Base':
                        reagents.append(component['amount']['moles']['value'])
                    elif input_type == 'metal and ligand':
                        catalysts.append(component['amount']['moles']['value'])
            
            for outcome in data.get('outcomes', []):
                for product in outcome.get('products', []):
                    if product.get('is_desired_product', False):
                        for measurement in product.get('measurements', []):
                            if measurement['type'] == 'YIELD':
                                yield_value = measurement['percentage']['value']
            
            # 计算分子指纹和描述符
            for smiles in smiles_list:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule:
                    fingerprint = GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
                    fingerprint_bits = list(fingerprint)
                    
                    # 计算分子描述符
                    descriptors = {desc_name: desc_func(molecule) for desc_name, desc_func in Descriptors.descList}
                    
                    # 将数据添加到列表中
                    data_list.append({
                        'fingerprint': fingerprint_bits,
                        'reagent': sum(reagents),
                        'catalyst': sum(catalysts),
                        'yield': yield_value,
                        **descriptors
                    })
                else:
                    print(f"Invalid SMILES: {smiles}")

# 将数据转换为 pandas 数据框
df = pd.DataFrame(data_list)

# 将分子指纹展开为单独的列
fingerprint_df = pd.DataFrame(df['fingerprint'].tolist())
df = df.drop(columns=['fingerprint']).join(fingerprint_df)

# 将所有列名转换为字符串类型
df.columns = df.columns.astype(str)

# 分离特征和目标变量
X = df.drop(columns=['yield'])
y = df['yield']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')