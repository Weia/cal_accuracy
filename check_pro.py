import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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


def cal_over_pro_and_save(model_result_file,pro_level,pointNum):
    #处理多个level和多个点
    for level in pro_level:
        print('deal_with_over_level:'+str(level))
        all_result = cal_over_pro(model_result_file, pointNum, level)
        for index, point in enumerate(pointNum):
            a_result = all_result[index]
            # print('result/' + str(point) + 'over_' + str(level) + '.txt')
            f_save = open('pro_result/' + str(point) + 'over_' + str(level) + '.txt', 'w')
            for i in range(len(a_result)):
                # print(a_result[i][0])
                f_save.write(a_result[i][0] + ' ')
                label = a_result[i][1].reshape(1, -1).tolist()[0]
                str_label = [str(x) for x in label]

                final_label = ' '.join(str_label)
                f_save.write(final_label)
                f_save.write(' '+'\n')

            f_save.close()



model_result_file='v1.1/520_result.txt'
# cal_over_pro(model_result_file,[0,1,2],0.5,0.6)
# check_point_pro(model_result_file)


pro_level=[i/10 for i in range(1,10)]

pointNum=[i for i in range(16)]
cal_over_pro_and_save(model_result_file,pro_level,pointNum)

