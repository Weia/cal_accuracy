import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv
#计算预测值与实际值的距离
def parse_pre_line(line):
    #解析根据预测概率产生的文件，pro_result里的
    list_line=line.split(' ')
    img_name=list_line[0].split('/')[-1]

    label=np.asarray([float(x) for x in list_line[1:-1]]).reshape(-1,3)
    return img_name,label


def parse_true_line(line):

    pass

def calculate_accuracy(label_path,true_label_path,pointNum,save_path='acc_result/acc_result.txt',del_over1000=True):
    """
    calculate the distance between the predict and ground truth
    this function only calculate one point file
    :param label_path: predict label path ,the form is img_path labels ...
    :param true_label_path:ground truth path,the form is imgName labels...
    :param FMapWidth:the width of the feature map which is the output of the model
    :param FMapHeight:the height of the feature map which is the output of the model
    :param save_path: the path to save the distance
    :return:None
    """

    #产生两个文件acc_result.txt and name.txt 保存到acc_result文件夹中。
    #acc_result.txt 距离文件
    #name.txt是距离超过1000的图片名称
    loss=[]

    #加载正确的label文件
    with open(true_label_path) as f_true_label:
        true_contents=f_true_label.readlines()

    #加载预测的label文件
    with open(label_path) as f_label:
        contents=f_label.readlines()
        for line in contents:
            # 图片名,预测label
            name,label=parse_pre_line(line)
            # print(name)
            pre_label=label[pointNum][0:-1]
            # print('pre',pre_label)
            for true_line in true_contents:

                list_true_line=true_line.split(' ')
                #找到对应的true label
                if name ==list_true_line[0]:
                    true_label=np.asarray([float(x) for x in list_true_line[1:-1]]).reshape(-1,2)[pointNum]
                    # print('true',true_label)
                    #计算距离,差，平方，和，开方sqrt((x1-x2^2）+(y1-y2)^2)
                    diff=true_label-pre_label
                    pingfang=np.power(diff,2)

                    he=np.add(pingfang[0],pingfang[1])
                    a_loss_result=np.sqrt(he)

                    # 暂且不计算可能是横向的图片,
                    if del_over1000:
                        if a_loss_result> 100:
                            with open('acc_result/name.txt', 'a+') as f:
                                f.write(name)
                                f.write('\n')
                            continue

                    str_loss=''

                    str_loss+=str(a_loss_result)
                    str_loss+=' '
                    str_loss+='\n'
                    loss.append(str_loss)
                    break

    #写入文件
    with open(save_path,'w') as f_save:
        print('写入文件%d'%(len(loss)))
        f_save.writelines(loss)

#计算每个点的平均loss
def cal_each_point_acc(acc_file_path,root_path='mean_loss_result/'):
    """
    calculate the loss of each point
    :param acc_file_path: the path save the loss of the predict
    :return: None
    """
    save_path=root_path+acc_file_path.split('/')[-1]
    print(save_path)
    with open(acc_file_path) as f_acc:
        contents=f_acc.readlines()
    n_samples=len(contents)
    print(n_samples)
    loss=[]
    for line in contents:
        content=[float(x) for x in line.split(' ')[:-1]]
        loss.append(content)

    arr_loss=np.asarray(loss)
    each_loss=np.mean(arr_loss,axis=0)
    print(each_loss)
    with open(save_path,'w+') as f:

        f.writelines(str(each_loss))



def checkout_high_loss_picture(images_file,path):
    #查看loss比较大的图片，从name.txt 中读取
    names=open(images_file).readlines()
    for name in names:
        img_path=path+'/'+name.replace('\n','')
        img=Image.open(img_path)
        plt.imshow(img)
        plt.show()



# checkout_high_loss_picture('foot_half/name.txt','/media/weic/新加卷/数据集/数据集/学生照片/test')


def statistic_each_level_loss(acc_file,save_file):
    """
    统计loss每个阶段的个数，以10为单位
    :param acc_file: 距离文件
    :param save_file: 保存到的文件，文件形式为csv
    :return:
    """
    contents=open(acc_file).readlines()
    result=np.zeros([11,1])

    for line in contents:
        list_line=np.asarray([float(x) for x in line.split(' ')[:-1]])
        for index,x in enumerate(list_line):
            level=int(x/10)
            # print(level)
            if level<10:#0-9 level 1-99 loss
                result[level][index]=result[level][index]+1


            else:#>100 loss
                result[10][index]+=1


    print(result)
    with open(save_file,'w') as f_save:
        writer=csv.writer(f_save)
        writer.writerows(result)
# statistic_each_level_loss('foot_half/acc_result.txt','result/statistic_result.csv')