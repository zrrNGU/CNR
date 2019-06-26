# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:31:33 2019

@author: zrr07
"""

import os
import tensorflow as tf 
from PIL import Image
import numpy as np
import h5py
train_data_dir = "png_for_train"
test_data_dir = "png_for_test"
char_set = "一二三四五六七八九十"

'''
writer=tf.python_io.TFRecordWriter('train_data.tfrecord')
for file_name in os.listdir(train_data_dir):
    image_path=os.path.join(train_data_dir,file_name)
    image= Image.open(image_path)
    image_raw=image.tobytes()
    for i in range(10):
        if file_name[0] == char_set[i]:
            example=tf.train.Example(features=tf.train.Features(feature={
            'lable':tf.train.Feature(int64_list=tf.train.Int64List(value=[i+1])) ,
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))                                                           
            }))
            writer.write(example.SerializeToString())
writer.close()

test_data_dir = "png_for_test"
writer=tf.python_io.TFRecordWriter('test_data.tfrecord')
for file_name in os.listdir(test_data_dir):
    image_path=os.path.join(test_data_dir,file_name)
    image= Image.open(image_path)
    image_raw=image.tobytes()
    for i in range(10):
        if file_name[0] == char_set[i]:
            example=tf.train.Example(features=tf.train.Features(feature={
            'lable':tf.train.Feature(int64_list=tf.train.Int64List(value=[i+1])) ,
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())
writer.close()
'''
def ImageToMatrix(file_name):
#    image_path=os.path.join(train_data_dir,file_name)
    im= Image.open(file_name)
    width,height = im.size
#    im = im.convert("L")
    data = im.getdata()
#    data = (np.array(data,dtype='float')-128)/128
    new_data = np.reshape(data,(height,width,3))
#    new_data = new_data.flatten()
    return new_data

train_set=[]
train_lable=[]
test_set=[]
test_lable=[]
classes=[1,2,3,4,5,6,7,8,9,10]
dict={"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
def get_files(file_dir):
    img_name=[]
    lables=[]
    for file_name in os.listdir(file_dir):
    #if file_name[0] == char_set[i]:
    #    data=ImageToMatrix(file_name)
    #    train_set.append(data)
        img_name.append(os.path.join(file_dir,file_name))
        lables.append(dict[file_name[0]])
    temp = np.array([img_name, lables])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    img_name = list(temp[:, 0])
    lables = list(temp[:, 1])
    lables = [int(i) for i in lables]
    
    return img_name,lables

image_list,train_lable = get_files(train_data_dir)
for file_name in image_list:
    train_set.append(ImageToMatrix(file_name))

image_list,test_lable = get_files(test_data_dir)
for file_name in image_list:
    test_set.append(ImageToMatrix(file_name))


f = h5py.File('data.h5', 'w')
f.create_dataset('X_train', data=train_set)
f.create_dataset('y_train', data=train_lable)
f.create_dataset('X_test', data=test_set)
f.create_dataset('y_test', data=test_lable)
f.create_dataset('list_classes', data=classes)
f.close()
