#泰坦尼克
import numpy as np
import torch
import pandas as pd

#读取文件，并把第一行作为列名,大坑：因为缺数据读取出来会变成一维，换成了pandas读取
#data = np.genfromtxt('train.csv',delimiter=',',dtype=None,names=True,filling_values=-1,skip_header=1)
df = pd.read_csv('train.csv',delimiter=',',header =0)

# 指定要删除的列名
cols_to_delete = ['Name', 'PassengerId','Ticket','Embarked','Cabin']

#删除列
#data = np.delete(data, cols_to_delete, axis=1)
df.drop(columns=cols_to_delete, inplace=True)

# 将 sex 列中的文本值映射为数S
df['Sex'] = df['Sex'].replace({'female': 0, 'male': 1})

# 计算平均年龄
mean_age = df['Age'].mean()

# 填充缺失值为平均年龄
df['Age'] = df['Age'].fillna(mean_age)

# 将DataFrame 中的数据转换为 NumPy 数
#只取结果列
survived = df.iloc[:, 0].values.astype('float32')
# 选择所有列，除了生存结果
x = df.iloc[:, 1:].to_numpy().astype('float32')

# 将 NumPy 数组转换为 Tensor，且为矩阵，类型为floatfloat32
x_data = torch.tensor(x)
y_data = torch.Tensor(survived).view(-1, 1)

#------------------读取数据---------------

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = torch.nn.Linear(6,4)
    self.linear2 = torch.nn.Linear(4,1)
    self.sigmoid = torch.nn.Sigmoid()
    self.relu = torch.nn.ReLU()
  def forward(self,x):
    x = self.sigmoid(self.linear1(x))
    x = self.sigmoid(self.linear2(x))
    return x
model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#继承dataset类
class DiabetesDataset(Dataset):
  def __init__(self):
    self.len = df.to_numpy().shape[0]
    self.x_data = x_data
    self.y_data = y_data
   #根据索引获得数据
  def __getitem__(self,index):
    return self.x_data[index],self.y_data[index]
  def __len__(self):
    return self.len
dataset = DiabetesDataset()
#初始化batchsizesize，shuffle是否打乱，num_workers:进程数
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle = True,
                         num_workers = 2)
#防止多线程出错
if __name__ == '__main__':
  for epoch in range(1000):
    for i,data in enumerate(train_loader,0):
      #准备数据
      inputs,labels = data
      y_pred = model(inputs)
      #计算损失
      loss = criterion(y_pred,labels)
      print(epoch,i,loss.item())
      #梯度清零
      optimizer.zero_grad()
      #反向传播并更新
      loss.backward()
      optimizer.step()

#-----------模型和minibatch------------

#测试
df_test = pd.read_csv('test.csv',delimiter=',',header =0)

# 指定要删除的列名
cols_to_delete = ['Name', 'Ticket','Embarked','Cabin']

# 删除序号，并保存到另一个变量
id = df_test.pop('PassengerId')

#删除列
df_test.drop(columns=cols_to_delete, inplace=True)

# 将 sex 列中的文本值映射为数S
df_test['Sex'] = df_test['Sex'].replace({'female': 0, 'male': 1})


# 填充缺失值为平均年龄
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())


# 将测试数据转化为 Tensor
test_data = df_test.to_numpy().astype('float32')
inputs = torch.tensor(test_data, dtype=torch.float32)

# 设置阈值
threshold = 0.5

# 对测试集进行预测
with torch.no_grad():
    outputs = model(inputs)
    preds = (outputs >= threshold).int()

#生成结果df
results = pd.DataFrame({
    'PassengerId': id,
    'Survived': preds.detach().numpy().flatten()
})

#转成csv文件
results.to_csv('titanic1.csv', index=False)


