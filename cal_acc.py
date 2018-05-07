
import numpy as np

#计算预测值与实际值的距离
def calculate_accuracy(label_path,true_label_path,FMapWidth,FMapHeight,save_path='./acc_result.txt'):
    """
    calculate the distance between the predict and ground truth
    :param label_path: predict label path ,the form is img_path labels ...
    :param true_label_path:ground truth path,the form is imgName labels...
    :param FMapWidth:the width of the feature map which is the output of the model
    :param FMapHeight:the height of the feature map which is the output of the model
    :param save_path: the path to save the distance
    :return:None
    """
    loss=[]

    #加载正确的label文件
    with open(true_label_path) as f_true_label:

        true_contents=f_true_label.readlines()

    #加载预测的label文件
    with open(label_path) as f_label:
        contents=f_label.readlines()
        for line in contents:
            content=line.split(' ')


            #图片名
            name=content[0].split('/')[-1]
            #预测label
            label=np.asarray([float(x) for x in content[1:-1]]).reshape(-1,2)

            for true_line in true_contents:


                list_true_line=true_line.split(' ')
                #找到对应的true label
                if name ==list_true_line[0]:
                    true_label=np.asarray([float(x) for x in list_true_line[1:-1]]).reshape(-1,2)

                    #计算距离,差，平方，和，开方sqrt((x1-x2^2）+(y1-y2)^2)
                    diff=true_label-label
                    pingfang=np.power(diff,2)
                    he=np.add(pingfang[:,0],pingfang[:,1])
                    a_loss_result=np.sqrt(he)
                    str_loss=''
                    for i in range(len(a_loss_result)):
                        str_loss+=str(a_loss_result[i])
                        str_loss+=' '
                    str_loss+='\n'

                    loss.append(str_loss)


    #写入文件
    with open(save_path,'w') as f_save:
        print('写入文件%d'%(len(loss)))
        f_save.writelines(loss)

#计算每个点的平均loss
def cal_each_point_acc(acc_file_path):
    """
    calculate the loss of each point
    :param acc_file_path: the path save the loss of the predict
    :return: None
    """
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





# label_path='/home/weic/project/cal_accuracy/test_file/result.txt'
# true_label='test_file/mirror_zero_label.txt'
# save_path=''
# Points=16
# ImageSize=64
#
# calculate_accuracy(label_path,true_label,64,64)


cal_each_point_acc('acc_result.txt')
