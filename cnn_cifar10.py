import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_confusion_matrix
import pdb

batch = 64
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

category = ('airplain','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

train_and_val_data = torchvision.datasets.CIFAR10(root="\data",download=True,train=True,transform=transform) # datasetを取得(Tensor型に変換して参照)

train_data, val_data = torch.utils.data.random_split(train_and_val_data,
                                                     [0.8,0.2])
test_data = torchvision.datasets.CIFAR10(root="\data",download=True,train=False,transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch,shuffle=True) # datasetを読み込む
val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=batch,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch,shuffle=False)



class Net(nn.Module): # 親クラスはnn.Module
    def __init__(self):
        super().__init__() #　nn.Moduleを呼び出す
        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,16,3)
        self.conv3 = nn.Conv2d(16,16,3)
        self.pool = nn.MaxPool2d(3,stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(16*22*22,80) # Affineレイヤ(weight,biasは ±(前層からの入力数)^(-1/2)の範囲の一様分布で初期化)
        self.fc2 = nn.Linear(80,80)
        self.fc3 = nn.Linear(80,10)

    def forward(self,x): # 順伝播
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))        
        x = x.view(x.size()[0], -1) # xの形を変える
        x = self.dropout(x)
        x = F.relu(self.fc1(x))# 行ベクトルに変形した入力xをAffineレイヤとReLUレイヤに通す
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) 
        x = self.dropout(x)
        x = self.fc3(x)     #　Affineレイヤ 結果は10個の実数

        return x

net = Net() # net によってクラスNet()を呼び出せるようにする
net.train() # optional

criterion = nn.CrossEntropyLoss() # 損失関数 誤差の算出方法
optimizer = optim.Adam(net.parameters(),lr=0.001) # 最適化の方法(勾配降下法) cf. Adam, AdamW

changes = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}


for epoch in range(50): # epochは10
    total_train_loss = 0
    train_ave_loss = 0
    total_val_loss = 0
    val_ave_loss = 0
    num_train = 0
    num_val = 0
    num_train_acc = 0
    num_val_acc = 0

    net.train()
    for train_x, train_label in train_loader:
        optimizer.zero_grad() # ミニバッチごとに勾配情報を0にリセット
        outputs = net(train_x)
        train_loss = criterion(net(train_x),train_label) # 出力と正解ラベルの誤差
        # loss: torch.Tensor, loss(number, grad) "loss.data -> number"
        train_loss.backward() # lossの勾配情報を計算
        optimizer.step() # 学習
        total_train_loss += train_loss.item()
        num_train += train_label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        num_train_acc += (predicted==train_label).sum().item()


    train_ave_loss = total_train_loss / num_train
    changes["train_loss"].append(train_ave_loss)
    changes["train_acc"].append(num_train_acc/num_train)

    net.eval
    for val_x, val_label in val_loader:
        optimizer.zero_grad()
        outputs = net(val_x)
        val_loss = criterion(net(val_x),val_label)
        val_loss.backward()
        total_val_loss += val_loss.item()
        num_val += val_label.size(0)
        _, predicted = torch.max(outputs.data, 1)
        num_val_acc += (predicted==val_label).sum().item()

    val_ave_loss = total_val_loss / num_val
    changes["val_loss"].append(val_ave_loss)
    changes["val_acc"].append(num_val_acc/num_val)




    if(epoch + 1)%1 == 0:
        print("Train loss per epoch:",epoch + 1,train_ave_loss)
        print("Validation loss per epoch:",epoch + 1,val_ave_loss)

#  - - - - - - - - - - - 
# 学習曲線（Loss Curve）の表示
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
y_train = changes["train_loss"]
x_train = list(range(len(y_train)))
ax1.plot(x_train,y_train,label = "Train", color = "red")
y_val = changes["val_loss"]
x_val = list(range(len(y_val)))
ax1.plot(x_val,y_val, label = "Validation", color = "blue")
ax1.set_xlabel("Number of epoches")
ax1.set_ylabel("Average Loss")
ax1.legend(loc = "upper right")

ax2 = fig.add_subplot(1,2,2)
y_train = changes["train_acc"]
x_train = list(range(len(y_train)))
ax2.plot(x_train,y_train, label = "Train", color = "red")
y_val = changes["val_acc"]
x_val = list(range(len(y_val)))
ax2.plot(x_val,y_val, label = "Validation", color = "blue")
ax2.set_xlabel("Number of epoches")
ax2.set_ylabel("Average Accuracy")
ax2.legend(loc = "upper right")

plt.show()


#  - - - - - - - - - - - 

net.eval() # パラメータを更新しない

with torch.no_grad(): # 以降Tensorの勾配情報を保持しない -> メモリ消費量削減
    correct = 0
    total = 0
    for train_x, train_label in train_loader:
        outputs = net(train_x)

        _, predicted = torch.max(outputs.data, 1) # torch.max(Tensor,1)で行の最大値とそのインデックスを取得。最大値は不要なので "_" に代入
        total += train_label.size(0)
        correct += (predicted == train_label).sum().item()

    train_accuracy = correct / total

with torch.no_grad():
    correct = 0
    total = 0
    predicted_labels = []
    test_labels = []
    for test_x, test_label in test_loader:
        outputs = net(test_x)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels.extend(predicted)
        test_labels.extend(test_label)
        total += test_label.size(0)
        # total += len(test_label)
        correct += (predicted == test_label).sum().item()

    test_accuracy = correct / total
    cmat_test = multiclass_confusion_matrix(input=torch.tensor(predicted_labels),
                                            target=torch.tensor(test_labels),
                                            num_classes=10)
print(cmat_test)
print(f"学習データの正解率：{100 * train_accuracy:.2f}%")
print(f"テストデータの正解率：{100 * test_accuracy:.2f}%")


net.eval()

predicted_labels = []
true_labels = []

with torch.no_grad():
    for i,(test_x, test_label) in enumerate(test_loader): # enumerateでインデックスを取得 -> iに代入
        output = net(test_x)
        _, predicted = torch.max(output, 1)
        predicted_labels.append(category[predicted[i].item()])
        true_labels.append(category[test_label[i]])

        if i == 9:
            break

print("予想：",predicted_labels[:10])
print("正解：",true_labels[:10])

num_images_per_row = 10

fig, axes = plt.subplots(nrows=1, ncols=num_images_per_row, figsize=(10, 4)) # 1行,10個の画像を表示

for idx, (images, labels) in enumerate(test_loader):
    if idx >= 1:
        break

    for i, image in enumerate(images):
        ax = axes[i]
        image = np.transpose(image,(1,2,0))
        image = image*0.5+0.5
        ax.imshow(image, cmap='viridis', vmin=0, vmax=1) # チャネル,高さ,幅 -> 高さ,幅,チャネル 
        ax.set_title(str(predicted_labels[i]))
        ax.axis('off')

        if i == num_images_per_row - 1:
            break

plt.tight_layout
plt.show()
fig.savefig("./output/cifar10_prediction.png")
