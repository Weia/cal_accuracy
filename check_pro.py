import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def is_full_people(label,less_pro):
    """
    根据头顶点、脚底点、左右腕点出现的概率，判断图片中人体是否完整
    :param label: (16,3)的标记值
    :less_pro float 低于这个概率的值认为没有检测到
    :return: True or False list [5],五个位置分别标记，头顶点、左右腕点、脚底点、人体，
    """
    index=[0,11,12,15]
    pro_list=label[index][:,2]
    result_list=[x>less_pro for x in pro_list]
    body=False if False in result_list else True
    result_list.append(body)

    return result_list


def check_point_pro(model_result_file):
    #查看点的概率和在图片中的位置
    with open(model_result_file) as f_result:
        contents=f_result.readlines()
    for line in contents:
        list_line=line.split(' ')
        img_path=list_line[0]
        label_pro=list_line[1:-1]
        arr_label=np.asarray([float(x) for x in label_pro]).reshape(-1,3)
        # print(arr_label)
        # img=Image.open(img_path)
        # plt.imshow(img)
        # plt.plot(arr_label[:,0],arr_label[:,1],'r+')
        # plt.show()

        # if arr_label[0,2]<0.3:
        #     img=Image.open(img_path)
        #     plt.imshow(img)
        #     plt.plot(arr_label[0,0],arr_label[0,1],'r+')
        #     plt.show()
def read_result_file(file_path):
    #读取result_file,txt文件,并返回文件内容
    try:
        with open(file_path) as f:
            contents=f.readlines()
    except FileNotFoundError :
        raise FileNotFoundError('not found the file : '+file_path)

    return contents


def parse_result_file_a_line(line):
    #解析load_model产生的文件
    list_line = line.split(' ')
    img_path = list_line[0]
    label_pro = list_line[1:-1]
    arr_label = np.asarray([float(x) for x in label_pro]).reshape(-1, 3)
    return img_path,arr_label


def cal_over_pro(model_result_file,pointNum,low_pro,high_pro=1.0):
    """
    保存某点预测值在某个概率区间的点的坐标和图像名称
    :model_result_file 保存模型预测值的文件
    :param pointNum: 第几个点，从0开始,是一个列表
    :param low_pro: 概率左区间
    :param high_pro: 概率右区间

    :return:
    """
    result=[list() for x in range(len(pointNum))]
    with open(model_result_file) as f_result:
        contents=f_result.readlines()
    for line in contents:
        img_path,arr_label=parse_result_file_a_line(line)
        for i in range(len(pointNum)):
            point_pro=arr_label[pointNum[i]][2]
            if point_pro>low_pro and point_pro<high_pro:
                result[i].append([img_path,arr_label])
    return result


def cal_over_pro_and_save(model_result_file,pro_level,pointNum,high_pro=1.0):
    #处理多个level和多个点
    for level in pro_level:
        print('deal_with_over_level:'+str(level)[0:3])
        all_result = cal_over_pro(model_result_file, pointNum, level,high_pro)
        for index, point in enumerate(pointNum):
            a_result = all_result[index]
            # print('result/' + str(point) + 'over_' + str(level) + '.txt')
            f_save = open('pro_result/' + str(point) + 'over_' + str(level)[0:3] + '.txt', 'w')
            for i in range(len(a_result)):
                # print(a_result[i][0])
                f_save.write(a_result[i][0] + ' ')
                label = a_result[i][1].reshape(1, -1).tolist()[0]
                str_label = [str(x) for x in label]

                final_label = ' '.join(str_label)
                f_save.write(final_label)
                f_save.write(' '+'\n')

            f_save.close()

def write_to_file(save_path,content):
    with open(save_path,'w') as f:
        for line in content:
            f.write(line)
    print('write to file %s success '%(save_path))



def show_img(img_path):
    #显示图片
    img=Image.open(img_path)
    plt.imshow(img)
    plt.show()



# pro_level=[float('-inf')]
#
# pointNum=[i for i in range(16)]
# cal_over_pro_and_save(model_result_file,pro_level,pointNum)

def main(less_pro):
    #将低于less_pro的图片认为是不完整图片并写入文件
    model_result_file = 'v1.1/520_result.txt'
    try:
        content = read_result_file(model_result_file)
    except Exception as info:
        print(info)
        exit()
    else:
        not_full_list = [list() for x in range(5)]  # 头顶点、左右腕点、脚底点、人体，
        root_save_path = 'not_full_image/'
        files_name = ['head.txt', 'left.txt', 'right.txt', 'foot.txt', 'body.txt']
        #less_pro = 0.1
        for line in content:
            imgName, labels = parse_result_file_a_line(line)
            is_full_list = is_full_people(labels, less_pro)
            # print(is_full_list)
            for index, tf in enumerate(is_full_list):
                if tf:
                    continue
                else:
                    not_full_list[index].append(line)
        for index, file_name in enumerate(files_name):
            save_path = root_save_path + str(less_pro) + file_name
            # print(save_path)
            write_to_file(save_path, not_full_list[index])


def show_not_full_image(not_full_image_path):
    content=read_result_file(not_full_image_path)
    for line in content:
        img_path,labels=parse_result_file_a_line(line)
        print(img_path)
        try:
            show_img(img_path)
        except FileNotFoundError :
            if '新加卷1' in img_path:
                img_path=img_path.replace('新加卷1','新加卷')
            else:
                img_path=img_path.replace('新加卷','新加卷1')
            try:
                show_img(img_path)
            except Exception as info:
                print(info)
                continue


# show_not_full_image('not_full_image/0.4body.txt')

# main(0.2)
# main(0.3)
# main(0.4)
# main(0.5)
# main(0.6)
# main(0.7)
# main(0.8)
# main(0.9)
# main(1.0)

