import numpy as np
import matplotlib.pyplot as plt
import os
import csv



def show_pro_loss(loss_path):
    with open(loss_path) as f_loss:
        contents=f_loss.readlines()
    new_contents=[]
    for line in contents:
        list_line=np.asarray([float(x) for x in line.split(' ')[:-1]])
        new_contents.append(list_line)

    np_contents=np.asarray(new_contents)
    np_less_40 = np.asarray([np_contents[index] for index,x in enumerate(np_contents[:, 0]) if x < 40])
    np_over_40=np.asarray([np_contents[index] for index,x in enumerate(np_contents[:, 0]) if x > 40])

    print(np_less_40.shape)
    print(np_over_40.shape)
    print(np_contents.shape)


    plt.plot(np_less_40[:,1],np_less_40[:,0],'g+')
    plt.plot(np_over_40[:,1],np_over_40[:,0],'r+')
    plt.vlines(0.6,0,1000)
    plt.vlines(0.7,0,1000)

    plt.vlines(0.8,0,1000)

    plt.vlines(0.9,0,1000)

    # for x in [60,70,80,90,100]:
    #     plt.plot([x, x], 'g-')
    plt.show()

def find_the_best_pro(loss_path):
    with open(loss_path) as f_loss:
        contents=f_loss.readlines()

    pro_range=np.arange(0.6,0.9,0.01)


    loss_range=np.arange(30,50,1)

    result=[[] for i in range(len(pro_range))]

    new_contents=[]
    for line in contents:
        list_line=np.asarray([float(x) for x in line.split(' ')[:-1]])
        new_contents.append(list_line)

    np_contents=np.asarray(new_contents)
    for index_pro,pro in enumerate(pro_range):
        for index_loss,loss in enumerate(loss_range):
            over_pro_over_loss_num=len([np_contents[index] for index,x in enumerate(np_contents)
                                    if (x[0]>loss and x[1]>pro)])
            over_pro_less_loss_num=len([np_contents[index] for index,x in enumerate(np_contents)
                                    if (x[0]<loss and x[1]>pro)])
            over_pro_mean=np.mean(np.asarray([np_contents[index] for index,x in enumerate(np_contents)
                                    if (x[1]>pro)])[:,0])
            over_pro_less_loss_mean=np.mean(np.asarray([np_contents[index] for index,x in enumerate(np_contents)
                                    if (x[0]<loss and x[1]>pro)])[:,0])
            less_pro_less_loss_num=len([np_contents[index] for index,x in enumerate(np_contents)
                                    if (x[0]<loss and x[1]<pro)])
            less_pro_over_loss_num=len([np_contents[index] for index,x in enumerate(np_contents)
                                    if (x[0]>loss and x[1]<pro)])

            result[index_pro].append([round(float(pro),2),loss,over_pro_over_loss_num,less_pro_over_loss_num,
                                          less_pro_less_loss_num,over_pro_less_loss_num,over_pro_mean,over_pro_less_loss_mean])

    # print(result)
    return result

# def get_

def ac_loss_get_result_column(result,loss,index):
    #获取指定loss在不同概率下的index位置上的值
    return result[:,0,0],result[:,loss-30,index]

def ac_pro_get_result_row(result,pro,index):
    #获取指定概率下的不同loss的index位置上的值
    return result[pro,:,1],result[pro,:,index]

def set_subplot_style(ax,x,y,title,point_style,x_label,y_label,toint=True,rotat=45,lab=''):
    ax.plot(x, y, point_style,label=lab)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if toint:
        for a, b in zip(x, y):
            plt.text(a, b, int(b), ha='center', va='bottom', fontsize=8,rotation=rotat)
    else:
        for a, b in zip(x, y):
            plt.text(a, b, round(b,4), ha='center', va='bottom', fontsize=8,rotation=rotat)



def draw_lines(result,loss,save_path):
    fig = plt.figure(figsize=(15, 10))

    #2 over_pro_over_loss_num
    ax1 = fig.add_subplot(321)
    x,y=ac_loss_get_result_column(result,loss,2)
    set_subplot_style(ax1,x,y,'over_pro_over_loss_num','p-','pro','num')

    #3 less_pro_over_loss_num
    ax2=fig.add_subplot(322)
    x2,y2=ac_loss_get_result_column(result,loss,3)
    set_subplot_style(ax2, x2, y2, 'less_pro_over_loss_num', 'b*-', 'pro', 'num')

    #4 less_pro_less_loss_num
    ax3 = fig.add_subplot(323)
    x3, y3 = ac_loss_get_result_column(result, loss, 4)
    set_subplot_style(ax3, x3, y3, 'less_pro_less_loss_num', 'gp-', 'pro', 'num')

    #5 over_pro_less_loss_num
    ax4 = fig.add_subplot(324)
    x4, y4 = ac_loss_get_result_column(result, loss, 5)
    set_subplot_style(ax4, x4, y4, 'over_pro_less_loss_num', 'co-', 'pro', 'num')

    #6 over_pro_mean, 大于概率的均值
    ax5 = fig.add_subplot(325)
    x5, y5 = ac_loss_get_result_column(result, loss, 6)
    set_subplot_style(ax5, x5, y5, 'over_pro_mean', 'mo-', 'pro', 'mean',False)


    #7over_pro_less_loss_mean
    ax6 = fig.add_subplot(326)
    x6, y6 = ac_loss_get_result_column(result, loss, 7)
    set_subplot_style(ax6, x6, y6, 'over_pro_less_loss_mean', 'yo-', 'pro', 'mean',False)

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, hspace=0.3, wspace=0.3)

    fig.savefig(save_path,dpi=300)

    # plt.show()

def cal_error_pro(over_pro_less_loss_num,over_pro_over_loss_num):
    # 计算错误率over pro 中less loss 占的比例
    total=over_pro_less_loss_num+over_pro_over_loss_num
    return over_pro_less_loss_num/total
def cal_over_pro_less_loss_in_total(over_less,over_over,less_less,less_over):
    #over less 占总样本的数量
    total=over_less+over_over+less_less+less_over
    return (over_less)/total



def draw_pro_line(result,loss,fig_save_path):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)

    for l in loss:
        x, over_over = ac_loss_get_result_column(result, l, 2)
        _, over_less = ac_loss_get_result_column(result, l, 5)
        _,less_over=ac_loss_get_result_column(result,l,3)
        _,less_less=ac_loss_get_result_column(result,l,4)
        y = cal_error_pro(over_less, over_over)
        y2=cal_over_pro_less_loss_in_total(over_less,over_over,less_less,less_over)
        set_subplot_style(ax1, x, y, 'less_loss_num in over pro num', 'p-', 'pro1', 'pro2',False,60,str(l))
        ax1.plot(x,y2,'o-',label=str(l))


    plt.legend()

    fig.savefig(fig_save_path,dpi=300)

    # plt.show()






loss_root_path='each_point_loss'
files=os.listdir(loss_root_path)
csv_root_path='analyze_result/find_pro'
fig_root_path='analyze_result/figure/prob'

for name in files:
    print(name)
    sta_result=find_the_best_pro(loss_root_path+'/'+name)
    # csv_save_path=csv_root_path+'/'+name.replace('txt','csv')
    # f_save=open(csv_save_path,'w')
    # writer=csv.writer(f_save)
    # writer.writerows(sta_result)
    # f_save.close()
    fig_save_path=fig_root_path+'/'+name.split('o')[0]+'.jpg'
    # print(fig_save_path)
    np_result=np.asarray(sta_result)
    draw_pro_line(np_result,[30,40,49],fig_save_path)





#读回csv文件
# f_csv=open('analyze_result.csv')
# content=csv.reader(f_csv)
# for line in content:
#     print([float(x) for x in line[0].replace(')','').replace('(','').split(',')])




