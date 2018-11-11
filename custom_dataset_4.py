'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? ***

    이제 training을 위한 코드를 작성해 보도록 하겠습니다.

    datasets으로 불러오는 방법을 수행해보죠

    우선 form torch.utils.data import DataLoader 명령으로

    DataLoader를 가져 옵니다.

    DataLoader에는 아까 만든 train_data를 넣어주고 몇가지 인자를 추가하여 값을 넣어줍니다.

    torch.utils.data.DataLoader가 입력을 받는 자세한 값들은 아래 링크에서 확인해보세요
    https://pytorch.org/docs/master/data.html?highlight=dataloader#torch.utils.data.DataLoader

    간단한 것들만 살펴보겠습니다.

    dataset     : torchvision.datasets.ImageFolder로 만들어낸 train_data값을 넣어주면 됩니다.
                  이어서 진행할 강의에서 사용할 torchvision.datasets.이하의 dataset도 불러온 다음  dataset = 하고 넣어주시면 됩니다.
                  사용방법은 아래를 참고하세요.

    batch_size  : batch_size는 말그대로 batch_size 입니다.
    shuffle     : dataset을 섞는 겁니다.
    num_worker  : 데이터 loader를 하기 위해 사용할 core의 수를 결정합니다. core가 많을 수록 빠릅니다.
    '''


from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
data=0

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,5)
        self.conv3 = nn.Conv2d(32,64,5)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 5),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(nn.Linear(32*6*6,10),
                                nn.ReLU(inplace=True),
                                nn.Linear(10,2))


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.layer1(x)
        print(x.shape)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x

if __name__ =="__main__":

    trans = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_data=torchvision.datasets.ImageFolder(root='C:\image\chair\\train',transform=trans)
    trainloader=DataLoader(dataset=train_data,batch_size=4,shuffle=True,num_workers=4)

    net = NN()

    for num, data in enumerate(trainloader):
        #print(num, type(data[0]), data[1])
        print(data[0].shape)
        out = net(data[0])
        print(out.shape)
        break
