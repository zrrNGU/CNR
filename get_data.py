# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:31:10 2019

@author: zrr07
"""

import os
import numpy as np
import struct
import PIL.Image

train_data_dir = "HWDB1.1trn_gnt"
test_data_dir = "HWDB1.1tst_gnt"

# 读取图像和对应的汉字
def read_from_gnt_dir(gnt_dir=train_data_dir):
	def one_file(f):
		header_size = 10
		while True:
			header = np.fromfile(f, dtype='uint8', count=header_size)
			if not header.size: break
			sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
			tagcode = header[5] + (header[4]<<8)
			width = header[6] + (header[7]<<8)
			height = header[8] + (header[9]<<8)
			if header_size + width*height != sample_size:
				break
			image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
			yield image, tagcode

	for file_name in os.listdir(gnt_dir):
		if file_name.endswith('.gnt'):
			file_path = os.path.join(gnt_dir, file_name)
			with open(file_path, 'rb') as f:
				for image, tagcode in one_file(f):
					yield image, tagcode

import scipy.misc
from sklearn.utils import shuffle

char_set = "一二三四五六七八九十"
 
def resize_and_normalize_image(img):
	# 补方
	pad_size = abs(img.shape[0]-img.shape[1]) // 2
	if img.shape[0] < img.shape[1]:
		pad_dims = ((pad_size, pad_size), (0, 0))
	else:
		pad_dims = ((0, 0), (pad_size, pad_size))
	img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
	# 缩放
	img = scipy.misc.imresize(img, (64 - 4*2, 64 - 4*2))
	img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
	assert img.shape == (64, 64)
 
	# img = img.flatten()
	# 像素值范围-1到1
	# img = (img - 128) / 128
	return img

train_counter=0
for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
	tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
	if tagcode_unicode in char_set:
		img = resize_and_normalize_image(image)
		# print(img)
		# print(tagcode_unicode)
		im = PIL.Image.fromarray(img)
		im.convert('RGB').save('png_for_train/' + tagcode_unicode + str(train_counter) + '.png')
		train_counter+=1
print(train_counter)

test_counter=0
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
	tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
	if tagcode_unicode in char_set:
		img = resize_and_normalize_image(image)
		# print(img)
		# print(tagcode_unicode)
		im = PIL.Image.fromarray(img)
		im.convert('RGB').save('png_for_test/' + tagcode_unicode + str(test_counter) + '.png')
		test_counter+=1
print(test_counter)