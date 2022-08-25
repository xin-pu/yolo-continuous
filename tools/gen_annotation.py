import os
import random
import time
from xml.etree.ElementTree import parse
import numpy as np
from tqdm import tqdm


def convert_annotation(annot_file, list_file):
    in_file = open(annot_file, encoding='utf-8')
    tree = parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xml_bbox = obj.find('bndbox')
        b = (int(float(xml_bbox.find('xmin').text)), int(float(xml_bbox.find('ymin').text)),
             int(float(xml_bbox.find('xmax').text)), int(float(xml_bbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


def gen_train_val_indexes_file(trainval_percent=0.9, train_percent=0.9):
    """
    用于在子数据集目录下生成按比例采样的索引文件
    :param trainval_percent:用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
    :param train_percent:用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
    """
    random.seed(0)
    if " " in os.path.abspath(dataset_root_folder):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    for ds in dataset_gather:
        print("Gen train val test Indexes for {}".format(ds))
        total_annot = []
        ds_annot_folder = os.path.join(dataset_root_folder, ds, "Annotations")
        temp_annot = os.listdir(ds_annot_folder)
        for xml in tqdm(temp_annot, colour='blue', ncols=100):
            if xml.endswith(".xml"):
                total_annot.append(xml)

        num = len(total_annot)
        indexes = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(indexes, tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        f_trainval = open(os.path.join(dataset_root_folder, ds, 'trainval.txt'), 'w')
        f_test = open(os.path.join(dataset_root_folder, ds, 'test.txt'), 'w')
        f_train = open(os.path.join(dataset_root_folder, ds, 'train.txt'), 'w')
        f_val = open(os.path.join(dataset_root_folder, ds, 'val.txt'), 'w')

        for i in indexes:
            name = total_annot[i][:-4] + '\n'
            if i in trainval:
                f_trainval.write(name)
                if i in train:
                    f_train.write(name)
                else:
                    f_val.write(name)
            else:
                f_test.write(name)

        f_trainval.close()
        f_train.close()
        f_val.close()
        f_test.close()
        print("Generate txt in ImageSets done.\r\n")


def gen_annotation():
    print("Start Generate Train and Val Annotations")
    time.sleep(0.2)
    for d1 in ['train', 'val', 'test', 'trainval']:
        out_file = os.path.join(dataset_root_folder, "{}.txt".format(d1))
        if os.path.exists(out_file):
            os.remove(out_file)
        for ds in dataset_gather:
            image_ids = open(os.path.join(dataset_root_folder, ds, '{}.txt'.format(d1)),
                             encoding='utf-8').read().strip().split()

            list_file = open(out_file, 'a', encoding='utf-8')
            for image_id in tqdm(image_ids, desc="write {} dataset to {}".format(ds, d1)):
                image_file = os.path.join(dataset_root_folder, ds, "JPEGImages", "{}.jpg".format(image_id))
                annot_file = os.path.join(dataset_root_folder, ds, "Annotations", "{}.xml".format(image_id))
                list_file.write(image_file)
                convert_annotation(annot_file, list_file)
                list_file.write('\n')
            list_file.close()


if __name__ == "__main__":
    dataset_root_folder = r'F:\PASCALVOC'
    dataset_gather = ['VOC2012', 'VOC2007']

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    nums = np.zeros(len(classes))
    # 不同数据集分别拆分为 trainval.txt,test.txt,train.txt,val.txt
    gen_train_val_indexes_file(trainval_percent=0.9, train_percent=0.9)

    # 聚合不同数据集
    gen_annotation()
