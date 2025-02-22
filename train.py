import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from resnet import ResNet18
#from mobilenetv1 import mobilenetv1_small
#from inceptionMolule import InceptionNetSmall
from base_resnet import resnet
#from resnetV1 import resnet as resnetV1
#from pre_resnet import pytorch_resnet18
from load_cifar10 import train_loader, test_loader
from SeResNet import se_resnet18
from SeVGGnet import Se_VGGNet
from Se_ResNet import Se_ResNet18
import os
import tensorboardX

if __name__ == '__main__':
#是否使用GPU
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 epoch_num = 200
 lr = 0.1
 batch_size = 128
 net=Se_ResNet18().to(device)

#loss
 loss_func = nn.CrossEntropyLoss()      #交叉熵损失,将得到的损失求平均值再输出,会输出一个数

#optimizer
# optimizer = torch.optim.Adam(net.parameters(), lr= lr)

 optimizer = torch.optim.SGD(net.parameters(), lr = lr,
                momentum=0.9, weight_decay=5e-4)

 scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=10,
                                            gamma=0.9)

 model_path = "models/Se_ResNet18"
 log_path = "logs/Se_ResNet18"
 if not os.path.exists(log_path):
    os.makedirs(log_path)
 if not os.path.exists(model_path):
    os.makedirs(model_path)
 writer = tensorboardX.SummaryWriter(log_path)


 step_n = 0
 for epoch in range(epoch_num):
    print(" epoch is ", epoch)
    net.train() #train BN dropout

    sum_loss_ = 0
    sum_correct_ = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, dim=1)

        correct = pred.eq(labels.data).cuda().sum()
        # print("epoch is ", epoch)
        # print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
        # print("train step", i, "loss is:", loss.item(),
        #       "mini-batch correct is:", 100.0 * correct / batch_size)
        sum_loss_ += loss.item()
        sum_correct_ += correct.item()

        writer.add_scalar("train loss", loss.item(), global_step=step_n)
        writer.add_scalar("train correct",
                          100.0 * correct.item() / batch_size, global_step=step_n)

        im = torchvision.utils.make_grid(inputs)
        writer.add_image("train im", im, global_step=step_n)

        step_n += 1

    train_loss_ = sum_loss_ * 1.0 / len(train_loader)
    train_correct_ = sum_correct_ * 100.0 / len(train_loader) / batch_size

    torch.save(net.state_dict(), "{}/{}.pth".format(model_path,
                                                     epoch + 1))
    with open('D:\kechenye\Se_ResNet18.txt', 'a') as f:
        print('Epoch is {}, train loss is: {} train correct is {}'.format(epoch+1,train_loss_,train_correct_), file=f, flush=True)

    print("epoch is", epoch + 1, "train loss is:", train_loss_,
         "train correct is:", train_correct_)

    scheduler.step()

    sum_loss = 0
    sum_correct = 0
    for i, data in enumerate(test_loader):
        net.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).cpu().sum()

        sum_loss += loss.item()
        sum_correct += correct.item()
        im = torchvision.utils.make_grid(inputs)
        writer.add_image("test im", im, global_step=step_n)

    test_loss = sum_loss * 1.0 / len(test_loader)
    test_correct = sum_correct * 100.0 / len(test_loader) / batch_size

    writer.add_scalar("test loss", test_loss, global_step=epoch + 1)
    writer.add_scalar("test correct",
                      test_correct, global_step=epoch + 1)

    with open('D:\kechenye\Se_ResNet18.txt', 'a') as f:
        print('Epoch is {}, test loss is: {} test correct is {}'.format(epoch+1,test_loss,test_correct), file=f, flush=True)

    print("epoch is", epoch + 1, "test loss is:", test_loss,
          "test correct is:", test_correct)

 writer.close()























