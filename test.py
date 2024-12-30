import os
from collections import defaultdict
from glob import glob
from sklearn.model_selection import train_test_split
import json
import random
# # 设置目标文件夹路径
# folder_path = 'MyData/UCMerced_LandUse/Images'
#
# # 获取文件夹下的所有文件和目录
# all_items = os.listdir(folder_path)
#
# # 筛选出子目录
# subdirectories = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]
#
# # 打印所有子目录
# print(subdirectories)

def align_landuse(root):
    classes = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
               'forest',
               'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass',
               'parkinglot',
               'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

    views = 4

    X = []
    Y = []

    for idx, c in enumerate(classes):
        items = glob(f"{root}/images/{c}/*.tif")
        sample = {}
        for i in range(len(items)//views):
            for j in range(views):
                sample[f"v{j}"]=items[i*views+j]

            X.append(sample)
            Y.append(idx)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    with open(f"{root}/train.json", 'w') as f:
        data = {
            'data': X_train,
            'labels': y_train
        }
        json.dump(data, f)

    with open(f"{root}/test.json", 'w') as f:
        data = {
            'data': X_test,
            'labels': y_test
        }
        json.dump(data, f)










if __name__=='__main__':
    align_landuse("E:\\Research\\Merge_Mrdd-share decoder\\MyData\\UCMerced_LandUse")


