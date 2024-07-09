
import torchvision.transforms
import numpy
import torch
import torch.nn as nn

class MotionDataset(torch.utils.data.Dataset):
    # Expect shape_one to be (frame,joint,dim)
    def __init__(self, path_file, shape_one, transform=None, meanstd=None):
        if path_file[-4:] == '.npy':
            self.npy = numpy.float32(numpy.load(path_file))
        if path_file[-4:] == '.npz':
            self.npy = numpy.float32(numpy.load(path_file)['S'])
        self.npy = self.npy[...,:shape_one[1],:shape_one[2]] # only get 19 points.
        print(self.npy.shape)
        self.shape_one = shape_one
        self.transform = transform

        if meanstd is None:
            self.orig_mean = numpy.mean(self.npy)
            self.orig_std = numpy.std(self.npy)
        else:
            self.orig_mean = meanstd[0]
            self.orig_std = meanstd[1]
        self.npy = (self.npy - self.orig_mean) / self.orig_std # normal=0 sd=1
        self.npy = self.npy * 0.5 + 0.5  # normal=0.5 sd=0.5

    def __len__(self):
        return self.npy.shape[0] - self.shape_one[0]

    def __getitem__(self, idx):
        npy = self.npy[idx:idx + self.shape_one[0]]
        if self.transform:
            npy = self.transform(npy)
        return npy

    def add_bias(self, npy):
        return (npy * self.orig_std) + self.orig_mean
    
class TransformToTensor(nn.Module):
    def __init__(self):
        super(TransformToTensor, self).__init__()

    def forward(self, x):
        return torch.tensor(x)

class TransformRandom(nn.Module):
    def __init__(self):
        super(TransformRandom, self).__init__()
        self.PI = torch.acos(torch.zeros(1)).item() * 2 # https://discuss.pytorch.org/t/np-pi-equivalent-in-pytorch/67157

    def forward(self, x):
        radius = torch.rand(1) * self.PI * 2
        rotation = torch.tensor([
            [torch.cos(radius), 0, torch.sin(radius)],
            [0, 1, 0],
            [-torch.sin(radius), 0, torch.cos(radius)],
        ])
        position = torch.randn(3)
        position[1] = 0
        return x @ rotation + position
    
def get_pair_loader(args):
    dataset = MotionDataset(args.path_dataset, args.shape_one, transform=torchvision.transforms.Compose([
        TransformToTensor(),
        TransformRandom(),
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader.count_item = len(dataset)

    if (not 'path_testset' in args) or (args.path_testset is None):
        return dataloader, None
    
    testset = MotionDataset(args.path_testset, args.shape_one, meanstd=(dataset.orig_mean, dataset.orig_std))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    testloader.count_item = len(testset)

    return dataloader, testloader