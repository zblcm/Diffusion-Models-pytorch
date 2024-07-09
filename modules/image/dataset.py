
import torchvision.transforms
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
    
class TransformSwapaxes(nn.Module):
    def __init__(self):
        super(TransformSwapaxes, self).__init__()

    def forward(self, x):
        return x.swapaxes(-3, -2).swapaxes(-2, -1)
    
def get_pair_loader(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.shape_one[:2], scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        TransformSwapaxes()
    ])

    dataset = torchvision.datasets.ImageFolder(args.path_dataset, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader.count_item = len(dataset)

    if (not 'path_testset' in args) or (args.path_testset is None):
        return dataloader, None
    
    testset = torchvision.datasets.ImageFolder(args.path_testset, transform=transforms)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    testloader.count_item = len(testset)

    return dataloader, testloader