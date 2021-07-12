import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 输入图片 shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),  # 卷积后的shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # 池化之后的shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层，输出为10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


class TrainMnistDataset:
    def __init__(self, epoch=1000, lr=0.001,  batch_size=50, data_root_dir="./datasets/", model_save_dir="./model_save_dir",
                 pretraind_model_path=False):
        self.data_root_dir = data_root_dir
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.model_save_dir = model_save_dir
        self.pretraind_model_path = pretraind_model_path

        if not os.path.exists(self.data_root_dir):
            os.mkdir(self.data_root_dir)
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        self.seed_set()
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    def seed_set(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def download_mnist_set(self):
        download_mnist = False
        if not (os.path.exists(self.data_root_dir)) or not os.listdir(self.data_root_dir):
            # not mnist dir or mnist is empyt dir
            download_mnist = True
        return download_mnist

    def plot_image(self, image, label):
        plt.imshow(image, cmap='gray')
        plt.title('%i' % label)
        plt.savefig("mnist-1.jpg")
        plt.show()

    def get_mnist_dataset(self):
        download_mnist_set = self.download_mnist_set()
        train_data = torchvision.datasets.MNIST(root=self.data_root_dir, train=True,
            transform=torchvision.transforms.ToTensor(),download=download_mnist_set)
        val_data   = torchvision.datasets.MNIST(root=self.data_root_dir, train=False,
            transform=torchvision.transforms.ToTensor())
        print("训练集图片和标签的shape: ", train_data.data.size(), train_data.targets.size())
        # torch.Size([60000, 28, 28]), torch.Size([60000])
        print("训练集图片和标签的shape: ", val_data.data.size(), val_data.targets.size())
        # torch.Size([10000, 28, 28]) torch.Size([10000])
        # self.plot_image(train_data.data[0].numpy(), train_data.targets[0])
        return train_data, val_data

    def pretrained_model_load(self, model):
        print('Loading weights into state dict...')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(self.pretraind_model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k])==np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def model_load(self):
        model = CNN()
        print(model)
        return model

    def save_train_information(self, message):
        path = os.path.join(self.model_save_dir, "train_info.txt")
        with open(path, 'a') as f:
            f.write(message)
            f.write("\n")

    def fit_one_epoch(self, model, train_loader, val_loader, epoch_size_train, epoch_size_val,
                               optimizer, loss_func, epoch_cur):

        # (一) 训练模型
        print("start training")
        model = model.train()  # Sets the module in training mode
        total_loss_train = 0
        for iteration, (images_batch, labels_batch) in enumerate(train_loader):
            # images_batch shape： torch.Size([50, 1, 28, 28]), labels_batch shape: torch.Size([50])
            # 1.1 判断在当前epoch下，是否遍历完训练集图片
            if iteration >= epoch_size_train:
                break

            # 1.2 进行no_grad和cuda加速
            images_batch = images_batch.cuda()
            labels_batch = labels_batch.cuda()
            assert images_batch.requires_grad==False, "images_batch.requires_grad should be set False "
            assert labels_batch.requires_grad==False, "labels_batch.requires_grad should be set False "

            # 1.3 前向传播与loss计算
            output = model(images_batch)  # output shape : (50, 10)
            loss = loss_func(output, labels_batch)  # output shape: (50, 10), labels_batch shape:(50,)

            # 1.4 之前梯度置零、反向传播计算当前梯度，权重更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 1.5 相关信息打印
            total_loss_train += loss.data.cpu().numpy()
            if iteration % 50 == 0:
                labels_batch_pred = torch.max(output, 1)[1].data.cpu().numpy()  # (50,)
                accuracy = float((labels_batch_pred == labels_batch.data.cpu().numpy()).astype(int).sum()) / float(self.batch_size)
                print(f"Epoch {(epoch_cur + 1):04d}/{self.epoch:04d} ||"
                      + f" 训练集【{(iteration + 1):04d}/{epoch_size_train:04d}】 ||"
                      + f" total_loss: {(total_loss_train / (iteration + 1)):.4f} ||"
                      + f" accuracy: {accuracy:.4f}")

        # （二） 验证模型
        print('Start Validation')
        total_loss_val = 0
        model = model.eval()  # Sets the module in eval mode
        for iteration, (images_batch, labels_batch) in enumerate(val_loader):
            # 2.1判断在当前epoch下，是否遍历完验证集图片
            if iteration >= epoch_size_val:
                break

            # 2.2 cuda加速和进行no_grad判断
            images_batch = images_batch.cuda()
            labels_batch = labels_batch.cuda()

            # 2.3 前向传播与loss计算
            output = model(images_batch)  # output shape : (50, 10)
            loss = loss_func(output, labels_batch)  # output shape: (50, 10), labels_batch shape:(50,)
            total_loss_val += loss.data.cpu().numpy()

            # 2.4 相关信息打印
            if iteration % 50 == 0:
                labels_batch_pred = torch.max(output, 1)[1].data.cpu().numpy()  # (50,)
                accuracy = float((labels_batch_pred == labels_batch.data.cpu().numpy()).astype(int).sum()) / float(
                    self.batch_size)
                print(f"Epoch {(epoch_cur + 1):04d}/{self.epoch:04d} ||"
                      + f" 验证集【{(iteration + 1):04d}/{epoch_size_val:04d}】 ||"
                      + f" total_loss: {(total_loss_val / (iteration + 1)):.4f} ||"
                      + f" accuracy: {accuracy:.4f}")

        # (三) 模型保存及相关信息保存
        print(f"train loss: {total_loss_train/epoch_size_train: .4f}|| val loss: {total_loss_val/epoch_size_val:.4f} ")
        info = f"Epoch{(epoch_cur+1):04d}-train_Loss{total_loss_train/(epoch_size_train):.4f}-Val_Loss{total_loss_val/(epoch_size_val+1):.4f}.pth"
        torch.save(model.module.state_dict(), os.path.join(self.model_save_dir, info))
        self.save_train_information(info)

    def train(self):
        # 1.获取数据集
        train_data, val_data = self.get_mnist_dataset()
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)  # (50, 1, 28, 28)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=True)

        # 2.模型加载与GPU加速
        model = self.model_load()
        if self.pretraind_model_path:
            model = self.pretrained_model_load(model)
        model = torch.nn.DataParallel(model).cuda()
        print("GPU加速了...赶快起飞吧..........!!!!!!!!!!   " * 3)

        # 3.设置优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_func = nn.CrossEntropyLoss()

        # 4.训练模型
        epoch_size_train = max(1, train_data.data.size()[0])//self.batch_size
        epoch_size_val   = max(1, val_data.data.size()[0])//self.batch_size
        for epoch_cur in range(self.epoch):
            self.fit_one_epoch(model, train_loader, val_loader, epoch_size_train, epoch_size_val,
                               optimizer, loss_func, epoch_cur)

def run():
    data_root_dir = "./mnist/"
    trian_mnist = TrainMnistDataset(data_root_dir=data_root_dir)
    trian_mnist.train()

if __name__ == "__main__":
    run()

