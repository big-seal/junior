import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# from model import *
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10("dataset", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# length看数据集有多少张
train_data_size = len(train_data)
test_data_size = len(test_data)
#字符串格式化
# 如果train_data_size=10,输出“训练集的长度为：10”
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

#用dataloader进行加载
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Ceshi(nn.Module):
    def __init__(self):
        super(Ceshi, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 ,32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32 ,5 ,1 ,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32 ,64 ,5 ,1 ,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024 ,64),
            nn.Linear(64 ,10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x

ceshi = Ceshi()
if torch.cuda.is_available():
    ceshi = ceshi.cuda()



# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()


# 定义优化器
#随机梯度下降
leaning_rate = 0.01
#1e-2=0.01
optimizer = torch.optim.SGD(ceshi.parameters(), lr=leaning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
#记录训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("train")




for i in range(epoch):
    print("----第{}轮训练开始----".format(i+1))
    #训练步骤开始
    # 这里会自动迭代数据集中的所有数据，每次抽取64个
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = ceshi(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        # 小批次64个数据优化一次
        optimizer.zero_grad()#梯度归零
        loss.backward()#计算梯度
        optimizer.step()#更新参数
        #
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}时，loss{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 此步不对梯度进行调优
    # 测试步骤开始
    ceshi.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = ceshi(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss为:{}".format(total_test_loss))
    print("整体测试集上的正确率为：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(ceshi, "ceshi_{}.pth".format(i))
    print("模型已保存")
writer.close()


































