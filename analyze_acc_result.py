import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
import os
import csv

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

def show_img(img_path,high_index_label):
    #显示图片
    img=Image.open(img_path)
    print(img.size)
    plt.imshow(img)
    plt.plot(high_index_label[:,0],high_index_label[:,1],'r*')
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

        try:
            show_img(img_path,high_loss_labels)
        except FileNotFoundError :
            if '新加卷1' in img_path:
                img_path=img_path.replace('新加卷1','新加卷')
            else:
                img_path=img_path.replace('新加卷','新加卷1')
            try:
                show_img(img_path,high_loss_labels)
            except Exception as info:
                print(info)
                continue



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


def cal_over_num_in_pro(acc_file,num,pro):

    with open(acc_file) as f_acc:
        contents = f_acc.readlines()
    with open('pro_result/'+acc_file.split('/')[-1]) as f_img:
        img_contents=f_img.readlines()
    pro_file='not_full_image/'+str(pro+0.1)[0:3]+'left.txt'
    with open(pro_file) as f_pro:
        pro_contents=f_pro.readlines()

    total =0
    for index,line in enumerate(contents):
        list_line = [float(x) for x in line.split(' ')[:-1]]
        if_over_list =[0 if x < num else 1 for x in list_line]
        if if_over_list[0]:
            label_line=img_contents[index]
            for pro_line in pro_contents:
                if pro_line==label_line:
                    total+=1
                    break
                else:
                    continue
        else:
            continue
    return total


def sta_index(index_path,loss_path):
    with open(index_path) as f_index:
        contents=f_index.readlines()

    result=np.zeros([16,1])

    for line in contents:
        list_line=[int(x) for x in line.split(' ')[:-1]]

        result[list_line]+=1
    return  result



# org_path='v1.1/520_result.txt'
# acc_path='v1.1/result386019/acc_result.txt'
# high_loss_imgs,high_loss_index=look_high_loss_img_size(org_path,acc_path,40)
# write_to_file('analyze_result/high_loss_over_40.txt',high_loss_imgs)
# str_high_loss_index=list_to_str(high_loss_index)
# write_to_file('analyze_result/index_over40.txt',str_high_loss_index)

# show_high_loss_file('result/high_loss.txt','result/index.txt')

# wh_list=statistic_img_width('/media/weic/新加卷/数据集/数据集/学生照片/test')
# for i in range(len(wh_list)):
#     print(len(wh_list[i]))
#     new=list_to_str(wh_list[i])
#     write_to_file('result/test_wh'+str(i)+'.txt',new)


# acc_file='v1.1/result386019/acc_result.txt'
# over_20=cal_a_point_loss_over_num(acc_file,40)
# print(over_20)

# show_high_loss_file('analyze_result/high_loss_over40.txt','analyze_result/index_over40.txt')

files=os.listdir('acc_result')

sta_result=np.zeros([1,10])
for name in files:
    pointNum=int(name.split('o')[0])
    str_level=name.split('_')[1][:3]
    level=int(float(str_level)*10) if str_level!='-in' else 10

    if (pointNum==11) and level==0:
        #print(pointNum in [0,9,10,15],pointNum)
        for i in range(10):
            pro=i*0.1
            print(pro)
            a=cal_over_num_in_pro('acc_result/'+name,40,pro)
            sta_result[0][i]=a
print(sta_result)
f=open('analyze_result/left_over_40.txt','w')
writer=csv.writer(f)
writer.writerows(sta_result)
f.close()
# cal_over_num_in_pro('acc_result/0over_0.0.txt',40,0.1)


# result=sta_index('analyze_result/index_over40.txt','analyze_result/high_loss_over_40.txt')
# print(result.tolist())


