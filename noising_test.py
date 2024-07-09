import torch
from ddpm import Diffusion
from modules.motion.dataset import get_dataloader
import argparse
import os
import numpy

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 128
args.shape_one = (64, 31, 3)
args.dataset_path = R"D:\Code\Projects\Github\Diffusion-Models-pytorch\data\temp.npy"

dataloader = get_dataloader(args)

diff = Diffusion(args.shape_one, device="cpu")

image = next(iter(dataloader))[0]
list_step = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 999]
t = torch.Tensor(list_step).long()

noised_image, _ = diff.noise_images(image, t)

print(noised_image.shape)
# save_image(noised_image.add(1).mul(0.5), "noise.jpg")

count_step = len(list_step)
path_dir = os.path.join("data", "noise_test", "frames")
os.makedirs(path_dir, exist_ok=True)
for index_step in range(count_step):
    step = list_step[index_step]
    image = noised_image[index_step,:,:,:]
    numpy.save(os.path.join(path_dir, "{0:05d}.npy".format(step)), image.to('cpu').numpy())

count_frame = noised_image.shape[1]
path_dir = os.path.join("data", "noise_test", "steps")
os.makedirs(path_dir, exist_ok=True)
for index_frame in range(count_frame):
    image = noised_image[:,index_frame,:,:]
    numpy.save(os.path.join(path_dir, "{0:05d}.npy".format(index_frame)), image.to('cpu').numpy())

image = image.to('cpu').numpy()
print(numpy.min(image), numpy.max(image))

noised_image = noised_image.to('cpu').numpy()
print(numpy.min(noised_image), numpy.max(noised_image))