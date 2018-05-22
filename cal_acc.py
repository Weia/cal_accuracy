
import numpy as np
import csv

#计算预测值与实际值的距离，16个点的距离，label形式不包括pro
def calculate_accuracy(label_path,true_label_path,save_path='result/acc_result.txt',del_over1000=False):
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
                    # 暂且不计算可能是横向的图片,
                    if del_over1000:
                        if a_loss_result[0] > 1000:
                            with open('name.txt', 'a+') as f:
                                f.write(name)
                                f.write('\n')
                            continue

                    str_loss=''
                    for i in range(len(a_loss_result)):
                        str_loss+=str(a_loss_result[i])
                        str_loss+=' '
                    str_loss+='\n'

                    loss.append(str_loss)
                    break


    #写入文件
    with open(save_path,'w') as f_save:
        print('写入文件%d'%(len(loss)))
        f_save.writelines(loss)

#计算每个点的平均loss
def cal_each_point_acc(acc_file_path,save_path='result/mean_loss.txt'):
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
    with open(save_path,'w+') as f:

        f.writelines(str(each_loss))




def statistic_each_level_loss(acc_file,save_file):
    contents=open(acc_file).readlines()
    result=np.zeros([11,16])


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


    #print(np.sum(result,axis=0))


# statistic_each_level_loss('result/acc_result.txt','result/statistic_result.csv')