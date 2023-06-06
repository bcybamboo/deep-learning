#########################
#-*-coding:utf-8-*-
import os
import os.path
import cv2
import numpy as np
from PIL import Image
import random

name_dict = {"bus":0, "family sedan":1, "fire engine":2, "heavy truck":3, "jeep":4,"minibus":5, "racing car":6, "SUV":7, "taxi":8, "truck":9}
data_root_path = "D:\kexie\dataset\car" # 数据集目录
test_file_path = data_root_path + "test.txt" # 测试集文件
train_file_path = data_root_path + "train.txt" # 训练集文件
eval_file_path=data_root_path + "val.txt" #验证集文件
name_data_list = {} # 记录每个类别图片 key:名称  value:路径列表

def save_train_test_file(path, name): # 将图片添加到字典
    if name not in name_data_list: # 该类别不在字典中
        img_list = []
        img_list.append(path) # 路径存入列表
        name_data_list[name] = img_list # 列表存入字典
    else:
        name_data_list[name].append(path) # 直接添加到列表

# 遍历每个子目录，将图片路径存入字典
dirs = os.listdir(data_root_path) # 列出数据集下的子目录
for d in dirs:
    full_path = data_root_path +'\\'+ d # 子目录完整路径
    if os.path.isdir(full_path): # 如果是目录
        imgs = os.listdir(full_path) # 列出子目录下的图片

        for img in imgs:
            img_full_path = full_path + "\\" + img # 图片路径
            if os.path.isdir(img_full_path):
                imgs_f=os.listdir(img_full_path)
                for i in imgs_f:
                    imgs_finish_path=img_full_path+'\\'+i
                    save_train_test_file(imgs_finish_path, d)  # 添加到字典
            else:
                save_train_test_file(img_full_path, d) # 添加到字典
    else: # 文件
        pass
with open(test_file_path, "w") as f:
    pass
with open(train_file_path, "w") as f:
    pass

# 遍历字典
for name, img_list in name_data_list.items():

    num = len(img_list) # 取出样本数量
    print("%s: %d张图像" % (name, num))

print("数据预处理完成.")

# 定义数据读取器
class dataset:
    def __init__(self, data_path, mode='train'):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param mode: train or eval
        """
        super().__init__()
        self.data_path = data_path
        self.img_paths = []
        self.labels = []

        if mode == 'train':
            with open(os.path.join(self.data_path, "train.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path,label= img_info.strip().split("   ")
                self.img_paths.append(img_path)
                #print(img_path)
                self.labels.append(int(label))
        elif mode == 'eval':
            with open(os.path.join(self.data_path, "val.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path,label= img_info.strip().split("   ")
                self.img_paths.append(img_path)
                #print(img_path)
                self.labels.append(int(label))

        else:
            with open(os.path.join(self.data_path, "test.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path= img_info.strip().split("\n")
                #print(img_path[0])
                self.img_paths.append(img_path[0])




    def __getitem__(self, index):

        # 第一步打开图像文件并获取label值
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        if np.random.random() > 0.5:
            img = np.fliplr(img)#镜像翻转
        img = np.array(img).transpose(2,0,1).astype("float")
        img=img/255.0#归一化
        label = self.labels[index]
        labels = np.zeros((1,9),dtype="int64")
        label = np.insert(labels,label,1)
        #print(label)

        return img, label

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t\t\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)

#生成索引
class Batchsample:
    def __init__(self,dataset=None,batch_size=16,shuffle=True):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.num_data=len(dataset)

        if self.num_data % batch_size==0:
            self.num_samples=self.num_data // batch_size
        else:
            self.num_samples=self.num_data // batch_size + 1
        indices=np.arange(self.num_data)

        if shuffle:
            np.random.shuffle(indices)
        self.indices=indices

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        batch_indices=[]
        for i in range(self.num_samples):
            if (i + 1) * self.batch_size <= self.num_data:
                for idx in range(i * self.batch_size, (i + 1) * self.batch_size):
                    batch_indices.append(self.indices[idx])
                yield batch_indices
                batch_indices = []
            else:
                for idx in range(i * self.batch_size, self.num_data):
                    batch_indices.append(self.indices[idx])
        if len(batch_indices) > 0:
            yield batch_indices
#一个数据生成器
class DataLoader:
    def __init__(self,dataset,sample=Batchsample,shuffle=True,batch_size=8):
        self.dataset=dataset
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.sample=sample
        self.samplers=self.sample().__iter__

    def __iter__(self):
        for sample_indices in self.samplers():
            img_list=[]
            label_list=[]
            for indice in sample_indices:
                img,label=self.dataset.__getitem__(indice)
                img_list.append(img)
                label_list.append(label)
            yield np.stack(img_list,axis=0),np.stack(label_list,axis=0)
            self.samplers = self.sample().__iter__

#训练数据加载
train_dataset =dataset(data_root_path,mode='train')
train_batchsample=Batchsample(train_dataset).__iter__

train_loader=DataLoader(train_dataset,train_batchsample,shuffle=True,batch_size=8)

#评估数据加载
eval_dataset =dataset(data_root_path,mode='eval')
eval_batchsample=Batchsample(eval_dataset).__iter__
eval_loader=DataLoader(eval_dataset,eval_batchsample,shuffle=True,batch_size=8)

#评估数据加载
test_dataset =dataset(data_root_path,mode='test')
test_batchsample=Batchsample(test_dataset).__iter__
test_loader=DataLoader(test_dataset,test_batchsample,shuffle=True,batch_size=8)

print("数据的预处理和加载完成！")




if __name__ == "__main__":

    for key, value in enumerate(train_loader):
        img,y = value
        print(key, img, y)

    '''for index,data in enumerate(eval_loader):
        ing,x = data
        print(ing,x.shape)'''

    '''for index,data in test_loader:
        print(index, data.shape)'''