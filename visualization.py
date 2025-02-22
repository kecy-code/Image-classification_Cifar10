import matplotlib.pyplot as plt

# 准备存储提取数据的列表
train_loss = []
test_loss =[]

#打开文本文件并读取每一行
with open(r'D:\kechenye\resnet18.txt', 'r') as file:
    for line in file:
        # 找到"Time=&"和"&mS"之间的字符串
        start_index = line.find("train loss is: ")
        end_index = line.find(" train correct", start_index)
        # start_index = line.find("&]=&")
        # end_index = line.find("&mS=", start_index)
        if start_index != -1 :
            # 提取loss数据
            loss_data = line[start_index + len("train loss is: "):end_index].strip()

            # 将loss数据转换为数字并添加到列表中
            try:
                loss_value = float(loss_data)
                train_loss.append(loss_value)
            except ValueError:
                # 数据转换失败时忽略错误
                continue

with open(r'D:\kechenye\resnet18.txt', 'r') as file:
    for line in file:
        # 找到"Time=&"和"&mS"之间的字符串
        start_index = line.find("test loss is: ")
        end_index = line.find(" test correct", start_index)
        # start_index = line.find("&]=&")
        # end_index = line.find("&mS=", start_index)
        if start_index != -1 :
            # 提取loss数据
            loss_data = line[start_index + len("test loss is: "):end_index].strip()

            # 将loss数据转换为数字并添加到列表中
            try:
                loss_value = float(loss_data)
                test_loss.append(loss_value)
            except ValueError:
                # 数据转换失败时忽略错误
                continue

train_loss_ = []
test_loss_ =[]
with open(r'D:\kechenye\VGGnet.txt', 'r') as file:
    for line in file:
        # 找到"Time=&"和"&mS"之间的字符串
        start_index = line.find("train loss is: ")
        end_index = line.find(" train correct", start_index)
        # start_index = line.find("&]=&")
        # end_index = line.find("&mS=", start_index)
        # if start_index != -1 and end_index != -1:
        if start_index != -1 :
            # 提取loss数据
            #loss_data = line[start_index + len("train loss is: "):end_index].strip()
            loss_data = line[start_index + len("train loss is: "): end_index].strip()
            # 将loss数据转换为数字并添加到列表中
            try:
                loss_value = float(loss_data)
                train_loss_.append(loss_value)
            except ValueError:
                # 数据转换失败时忽略错误
                continue

with open(r'D:\kechenye\VGGnet.txt', 'r') as file:
    for line in file:
        # 找到"Time=&"和"&mS"之间的字符串
        start_index = line.find("test loss is: ")
        end_index = line.find(" test correct", start_index)
        # start_index = line.find("&]=&")
        # end_index = line.find("&mS=", start_index)
        if start_index != -1 :
            # 提取loss数据
            loss_data = line[start_index + len("test loss is: "):end_index].strip()

            # 将loss数据转换为数字并添加到列表中
            try:
                loss_value = float(loss_data)
                test_loss_.append(loss_value)
            except ValueError:
                # 数据转换失败时忽略错误
                continue



'''
print(train_loss[:])
print(len(train_loss))
print(test_loss[:])
print(len(test_loss))
'''
x=list(range(1,201))
#plt.plot(x, test_loss, color='red',label='test correct',linewidth=1,linestyle="solid",label = 'SeVGGnet')
# 可以在一个画布上绘制多张图片
plt.plot(x, train_loss, color='green',linewidth=1,linestyle="solid",label='ResNet18')
plt.plot(x, test_loss, color='green',linewidth=1,linestyle="solid",label='ResNet18')
plt.plot(x, train_loss_, color='red',linewidth=1,linestyle="solid",label='VGGnet')
plt.plot(x, test_loss_, color='red',linewidth=1,linestyle="solid",label='VGGnet')
# 可以在一个画布上绘制多张图片
#plt.plot(x, test_loss_, color='red',linestyle='-',label='Se-ResNet18')
#plt.plot(x, test_loss_, color='green',label='train correct',linewidth=1,linestyle="solid")
# ax = plt.axes()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)


y_major_locator=plt.MultipleLocator(5)#以每3显示
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)

#plt.legend(loc="lower right")
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('ResNet18 && VGGnet')
plt.show()



