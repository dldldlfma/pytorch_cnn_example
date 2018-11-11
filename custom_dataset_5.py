'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? ***

    training을 위한 코드를 작성 중 optim과 loss를 결정하고 학습을 진행해 보겠습니다.


    '''


from torch.utils.data import DataLoader
import torch

import torchvision
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
data=0

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        """
        self.conv1 = nn.Conv2d(3,16,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,5)
        self.conv3 = nn.Conv2d(32,64,5)
        self.fc1 = nn.Linear(64*9*9, 512)
        self.fc2 = nn.Linear(512,2)
        """

        self.layer = nn.Sequential(nn.Conv2d(3,6,5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(6,16,5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.fc = nn.Sequential(nn.Linear(400,120),
                                nn.ReLU(),
                                nn.Linear(120,84),
                                nn.ReLU(),
                                nn.Linear(84, 2)
                                )



    def forward(self,x):
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x=x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        """

        x=self.layer(x)

        x=x.view(x.shape[0],-1)

        x=self.fc(x)
        return x


if __name__ =="__main__":

    if (torch.cuda.is_available() ==1):
        print("cuda is available")
        device ='cuda'
    else:
        device = 'cpu'

    #device = 'cpu'

    trans = transforms.Compose([
        transforms.Resize((40 ,40)),
        transforms.RandomCrop((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data=torchvision.datasets.ImageFolder(root='C:\image\chair\\train',transform=trans)
    trainloader=DataLoader(dataset=train_data,batch_size=8,shuffle=True,num_workers=4)
    length = len(trainloader)
    print(length)

    net = NN().to(device)
    optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    epochs =30
    for epoch in range(epochs):
        running_loss=0.0
        for num, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = net(inputs)

            loss = loss_function(out, labels)
            loss.backward()
            optim.step()

            running_loss +=loss.item()
            if num % length == (length-1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, num + 1, running_loss / length))
                running_loss = 0.0
