import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
import os

import cal_acc
import cal_a_point


def main():
    label_path = 'v1.1/521_result.txt'
    true_label_path = 'test.txt'

    cal_acc.calculate_accuracy(label_path, true_label_path, del_over1000=False)

    cal_acc.cal_each_point_acc('result/acc_result.txt')

    cal_acc.statistic_each_level_loss('result/acc_result.txt','result/statistic_result.csv')

def cal_a_point_loss(true_label_path,pointNum=0,level=0.9,cal_all=False):
    root_path='pro_result'
    files=os.listdir(root_path)
    if cal_all:
        for file in files:
            label_path = root_path + '/' + file
            save_path='acc_result/'+file
            cal_a_point.calculate_accuracy(label_path, true_label_path, pointNum, save_path=save_path)


    else:
        for file in files:
            pNum=int(file[0])
            lev=float(file.split('_')[1][0:3])

            if pNum==pointNum and lev==level:
                label_path = root_path + '/' + file
                save_path='acc_result/'+str(pNum)+'_'+str(lev)+'_result.txt'
                cal_a_point.calculate_accuracy(label_path,true_label_path,pointNum,save_path=save_path)





def analyze_csv(csv_path):
    #统计0-30的loss的点的个数

    f_csv=open(csv_path)
    reader=csv.reader(f_csv)
    less_30=[]

    for index,line in enumerate(reader):
        if index<3:
            list_line=[float(x) for x in line]
            less_30.append(list_line)
        else:
            break

    print(len(less_30))
    arr_less_30=np.asarray(less_30)
    sum_less_30=np.sum(arr_less_30,axis=0)
    print(sum_less_30)



if __name__ == '__main__':
    # print('386019')
    # analyze_csv(csv_path='v1.1/result386019/statistic_result.csv')
    # print('575319')
    # analyze_csv('v1.1/result575319/statistic_result.csv')
    true_label_path = 'test.txt'
    cal_a_point_loss(true_label_path,cal_all=True)
    # cal_a_point.cal_each_point_acc('acc_result/3_0.9_result.txt')


# with open('v0.0/result/name.txt') as f:
#     contents=f.readlines()
# with open('v0.0/test.txt') as f_test:
#     lines=f_test.readlines()
#     for line in contents:
#         name=line.replace('\n','')
#         img_path=r'/media/weic/新加卷/数据集/数据集/学生照片/test'+'//'+name
#         for label in lines:
#             list_label=label.split(' ')
#             test_name=list_label[0]
#             if name==test_name:
#                 labels=np.asarray([float(x) for x in list_label[1:-1]]).reshape(-1,2)
#
#                 img = Image.open(img_path)
#                 plt.imshow(img)
#                 plt.plot(labels[:,0],labels[:,1],'r+')
#                 plt.show()


