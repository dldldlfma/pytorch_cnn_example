'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? ***

    상황 설명 =>  저는 빨간색의자와 회색의자를 구분하고 싶어요!
    그런데 사진을 어떻게 가져오고 어떻게 입력해야 되는지 잘 모르겠어요

    torchvision.datasets.ImageFolder는 내가 가지고 있는 사진을 이용하는 방법입니다.

    가지고 있는 사진을 아래와 같이 구분해 두고 시작합니다.

    C:\ 드라이브에
    image라는 폴더를 만들고 (=> C;\image)
    구분할 내용을 입력합니다. 의자니까 chair라고 하죠(=> C:\image\chair)
    chiar폴더 안에 train과 test 폴더를 만듭니다. (=>C:\image\chair\train & C:\image\chair\test)
    train폴더 안에 구분할 의자별 폴더를 만듭니다. (=>C:\image\chair\train\red & C:\image\chair\train\gray)
    이제 각 폴더에 색깔별로 넣으면 됩니다.

    C:\
    |
    --image
        |
        ----chiar
              |
              ----train
                   |-----red
                   |-----gray
              ----test
                   |-----red
                   |-----gray


    torchvision.datasets.ImageFolder는 다음과 같은 내용을 인자로 받습니다.

    root                          = 내 폴더의 위치를 str 값으로 입력 받음
    transform(optional)           = 입력받을 데이터들을 원하는 형태로 수정하는 방법입니다.
                                    torch는 입력 값이 무조건 tensor여야 하는데 여기서 하면 되겠지요?
                                    모르시겠다면 앞에 xx강을 참조하세요!
    target_transform(optional)    = A function/transform that takes in the target and transforms it.
    loader                        = A function to load an image given its path.
    '''




import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL

import torchvision

data=0

if __name__ =="__main__":

    train_data=torchvision.datasets.ImageFolder(root='C:\image\chair\\train',transform=None)

    for num, value in enumerate(train_data):
        data, label = value
        # data.show()로 사진을 직접 볼수 있습니다. 그런데 이건 한번에 다열리니까 break와 함께 쓰기!
        #data.show()
        #break

        print(num, data, label)



