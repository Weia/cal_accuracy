import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
import os

def look_high_loss_img_size(org_path,acc_path,loss):
    #返回高于loss的图片信息，imgpath和labels
    with open(acc_path) as f_acc:
        acc_contents=f_acc.readlines()
    with open(org_path) as f_org:
        acc_org=f_org.readlines()
    high_loss_img_path=[]
    high_loss_point_index=[]
    for index,line in enumerate(acc_contents):
        list_line=np.asarray([float(x) for x in line.split(' ')[:-1]])

        over_100=[x>loss for x in list_line]
        high_loss_index=[index for index,x in enumerate(over_100) if x]
        have_high_loss=True if True in over_100 else False
        if have_high_loss:
            high_loss_img_path.append(acc_org[index])
        if len(high_loss_index):
            high_loss_point_index.append(high_loss_index)

        assert len(high_loss_point_index)==len(high_loss_img_path),'point_index num not equal img_path'

    print('find over %d loss %d lines'%(loss,len(high_loss_img_path)))
    return high_loss_img_path,high_loss_point_index


def write_to_file(save_path,content):
    with open(save_path,'w') as f:
        for line in content:
            f.write(line)
    print('write to file %s success '%(save_path))

def list_to_str(list):
    new_list=[]
    for line in list:
        new_list.append(' '.join([str(x) for x in line])+' \n')
    return new_list


def parse_result_file_a_line(line):
    #解析load_model产生的文件
    list_line = line.split(' ')
    img_path = list_line[0]
    label_pro = list_line[1:-1]
    arr_label = np.asarray([float(x) for x in label_pro]).reshape(-1, 3)
    return img_path,arr_label

def show_img(img_path):
    #显示图片
    img=Image.open(img_path)
    print(img.size)
    plt.imshow(img)
    plt.show()

def show_high_loss_file(high_loss_file_path,index_path):
    with open(high_loss_file_path) as f_high:
        contents=f_high.readlines()
    with open(index_path) as f_index:
        index_contents=f_index.readlines()
    for index,line in enumerate(contents):
        img_path,labels=parse_result_file_a_line(line)
        # print(labels)
        label_index=[int(x) for x in index_contents[index].split(' ')[:-1]]
        # print(label_index)
        high_loss_labels=labels[label_index]
        print('index',index_contents[index])
        print('概率值：',high_loss_labels[:,2])
        show_img(img_path)



def statistic_img_width(img_root_path):
    imgs=os.listdir(img_root_path)
    wh_list=[list() for i in range(6)]
    for index,name in enumerate(imgs):
        print(index)
        img_path=img_root_path+'/'+name
        try:
            img = Image.open(img_path)
        except Exception as info:
            print(info)
            continue
        width,height=img.size
        wh_list[int(width/1000)].append([width,height])

    return wh_list


def cal_a_point_loss_over_num(acc_file,num):
    with open(acc_file) as f_acc:
        contents=f_acc.readlines()
    total=np.zeros([1,16])
    for line in contents:
        list_line=[float(x) for x in line.split(' ')[:-1]]
        if_over_list=np.asarray([0  if x <num else 1 for x in list_line]).reshape(1,16)
        total+=if_over_list

    return total.tolist()







# org_path='v1.1/520_result.txt'
# acc_path='v1.1/result386019/acc_result.txt'
# high_loss_imgs,high_loss_index=look_high_loss_img_size(org_path,acc_path,100)
# write_to_file('result/high_loss.txt',high_loss_imgs)
# str_high_loss_index=list_to_str(high_loss_index)
# write_to_file('result/index.txt',str_high_loss_index)

# show_high_loss_file('result/high_loss.txt','result/index.txt')

# wh_list=statistic_img_width('/media/weic/新加卷/数据集/数据集/学生照片/test')
# for i in range(len(wh_list)):
#     print(len(wh_list[i]))
#     new=list_to_str(wh_list[i])
#     write_to_file('result/test_wh'+str(i)+'.txt',new)


acc_file='v1.1/result386019/acc_result.txt'
over_20=cal_a_point_loss_over_num(acc_file,40)
print(over_20)