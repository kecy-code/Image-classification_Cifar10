import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

import glob
import numpy as np
import cv2
import os

train_list = glob.glob("D:\DeepLearning\pythonProject2-Cifar10\cifar-10-batches-py/test_batch*")
print(train_list)
save_path = "D:\DeepLearning\pythonProject2-Cifar10\cifar-10-batches-py\TEST"

for l in train_list:
    print(l)
    l_dict = unpickle(l)
    # print(l_dict)
    print(l_dict.keys())

    for im_idx, im_data in enumerate(l_dict[b'data']):
        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]
        print(im_label, im_name, im_data)
        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))

        # cv2.imshow("im_data",cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_path,
                                             im_label_name)):
            os.mkdir("{}/{}".format(save_path, im_label_name))

        cv2.imwrite("{}/{}/{}".format(save_path,
                                      im_label_name,
                                      im_name.decode("utf-8")), im_data)



































# data_list = glob.glob("/home/kuan/dataset/"
#                       "cifar-11-batches-py/data_batch*")
# for path in data_list:
#     data = unpickle(path)
#     for i in range(len(data[b"labels"])):
#
#         im_data = np.reshape(data[b"data"][i], (3, 32, 32))
#         im_data = np.transpose(im_data, (1, 2, 0))
#         im_name = data[b'filenames'][i].decode("utf-8")
#         im_label = label_name[data[b"labels"][i]]
#
#         if not os.path.exists("/home/kuan/dataset/cifar-11-batches-py/train/{}"
#                                   .format(im_label)):
#             os.mkdir("/home/kuan/dataset/cifar-11-batches-py/train/{}"
#                                   .format(im_label))
#
#         cv2.imwrite("/home/kuan/dataset/cifar-11-batches-py/train/{}/{}"
#                                   .format(im_label, im_name), im_data)
