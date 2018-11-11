'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? ***

    2. 읽어온 Custom Dataset에 대해서 transform 진행하기

    from torchvision import transoforms 추가

    transforms는 입력된 이미지를 transform 할 수 있는 함수들이 모여 있습니다.

    transforms를 통해 이미지의 크기와 type을 변경해 봅시다.

    transforms.Compose( ) 를 이용해서 변형을 한번에 여러개를 진행할 수 있습니다.

    transforms.Compose( 이 안에 list 형태로 transforms 명령어 들을 넣어주면 됩니다. )

    ex)
    trans = transforms.Compose( [transforms.Resize((256,256)),
                                 transforms.ToTensor()]
     )

    '''


import torchvision
from torchvision import transforms

data=0

if __name__ =="__main__":

    #실제로 예를 들기 위해서 Resize만 진행했습니다.
    #다음장에서는 ToTensor까지 적용된 코드로 수행하도록 하겠습니다.

    trans = transforms.Compose([
        transforms.Resize((256,256)),
    ])
    train_data=torchvision.datasets.ImageFolder(root='C:\image\chair\\train',transform=trans)

    for num, value in enumerate(train_data):
        data, label = value
        print(num, data, label)
        break

data.show()