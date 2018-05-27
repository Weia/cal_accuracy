import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
import os

import cal_acc
import cal_a_point


def main():
    label_path = 'final_result/final_test_result.txt'
    true_label_path = 'test.txt'

    cal_acc.calculate_accuracy(label_path, true_label_path, del_over1000=False)

    cal_acc.cal_each_point_acc('result/acc_result.txt')

    cal_acc.statistic_each_level_loss('result/acc_result.txt','result/statistic_result.csv')

def cal_a_point_loss(true_label_path,pointNum=0,level=0.9,cal_all=False):
    root_path='pro_result'
    files=os.listdir(root_path)

    for file in files:

        label_path = root_path + '/' + file
        save_path = 'acc_result/' + file
        pNum = int(file.split('o')[0])
        lev = float('-inf') if file.split('_')[1][0:3]=='-in' else float(file.split('_')[1][0:3])
        print(file, pNum, lev)
        if cal_all:
            cal_a_point.calculate_accuracy(label_path, true_label_path, pNum, save_path=save_path)
        else:
            if pNum==pointNum and lev==level:
                label_path = root_path + '/' + file
                save_path='acc_result/'+file
                cal_a_point.calculate_accuracy(label_path,true_label_path,pointNum,save_path=save_path)
                break


def cal_every_level_loss_over_num(acc_path,num):

    with open(acc_path) as f_acc:
        contents = f_acc.readlines()


    total =0
    for line in contents:
        list_line = [float(x) for x in line.split(' ')[:-1]]
        if_over_list =[0 if x < num else 1 for x in list_line]
        total += if_over_list[0]


    return total




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
    #
    cal_a_point_loss(true_label_path,cal_all=True)
    # acc_files=os.listdir('acc_result')
    # final_mean_result=np.zeros([16,11])
    #
    # for file in acc_files:
    #     num=int(file.split('o')[0])
    #
    #     level=10 if file.split('_')[1][0:3]=='-in' else int(file.split('_')[1][2:3])
    #
    #     a_point_mean_loss=cal_a_point.cal_each_point_acc('acc_result/'+file)
    #     final_mean_result[num][level]=a_point_mean_loss[0]
    # with open('mean_loss_result/mean_loss.csv','w') as f :
    #     writer=csv.writer(f)
    #     writer.writerows(final_mean_result)



    # files=os.listdir('acc_result')
    #
    # sta_result=np.zeros([16,11])
    # for name in files:
    #     pointNum=int(name.split('o')[0])
    #     str_level=name.split('_')[1][:3]
    #     level=int(float(str_level)*10) if str_level!='-in' else 10
    #     print(pointNum)
    #     a=cal_every_level_loss_over_num('acc_result/'+name,40)
    #     sta_result[pointNum][level]=a
    # print(sta_result)
    # f=open('analyze_result/level_over_40.txt','w')
    # writer=csv.writer(f)
    # writer.writerows(sta_result)
    # f.close()








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


