
import os
import os.path
import cv2
import numpy as np
from PIL import Image
import random

import net

def write_txt(content, filename,mode='w'):
    """
    把路径＋标签写进文本
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以3个空格作为分隔符，与名字相分开
                    str_line = str_line + str(data) + "   "

                else:
                    # 每行最后一个数据换行符
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def get_files_list(dir):
    '''
    实现遍历训练集和验证集所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            #print("parent is: " + parent)
            #print("filename is: " + filename)
            #print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            curr_file=parent.split(os.sep)[-1]
            if curr_file=='bus':
                labels=0
            elif curr_file=='family sedan':
                labels=1
            elif curr_file=='fire engine':
                labels=2
            elif curr_file=='heavy truck':
                labels=3
            elif curr_file=='jeep':
                labels=4
            elif curr_file=='minibus':
                labels=5
            elif curr_file=='racing car':
                labels=6
            elif curr_file=='SUV':
                labels=7
            elif curr_file=='taxi':
                labels=8
            elif curr_file=='truck':
                labels=9
            files_list.append([dir+'\\'+os.path.join(curr_file, filename),labels])
    return files_list

def get_test_files_list(dir):
    '''
    实现遍历测试集所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            files_list.append(dir+"\\"+os.path.join(filename))
    return files_list

def write_test_txt(content, filename, mode='w'):
    """
    把路径＋标签写进文本
    """
    with open(filename, mode) as f:
        for line in content:
            f.write(str(line)+"\n")


if __name__ == '__main__':
    train_dir = "D:\kexie\dataset\car\\train"
    train_txt="D:\kexie\dataset\car\\train.txt"
    train_data = get_files_list(train_dir)
    write_txt(train_data,train_txt,mode='w')

    val_dir = "D:\kexie\dataset\car\\val"
    val_txt="D:\kexie\dataset\car\\val.txt"
    val_data = get_files_list(val_dir)
    write_txt(val_data,val_txt,mode='w')

    test_dir = "D:\kexie\dataset\car\\test"
    test_txt = "D:\kexie\dataset\car\\test.txt"
    test_data = get_test_files_list(test_dir)
    write_test_txt(test_data, test_txt, mode='w')

# 根据文件路径和文件名读取相应数据
def readImg(imgPath):
    imgs = []
    for item in imgPath:
        img = cv2.imread(item)
        imgs.append(img)
    return np.array(imgs)