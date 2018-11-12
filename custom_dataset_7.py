import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision

from model import model


if __name__ == "__main__":

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,128)),
        torchvision.transforms.ToTensor()
    ])
    test_data = torchvision.datasets.ImageFolder(root='./origin_data', transform=trans)

    testloader = DataLoader(dataset=test_data, batch_size=8, shuffle=True, num_workers=4)

    pre_train_net = model.NN()
    pre_train_net.load_state_dict(torch.load('./model/model.pth'))

    device='cuda'
    pre_train_net = pre_train_net.to(device)

    for num, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.to(device)
        out = pre_train_net(inputs)
        print(out,labels)
        break