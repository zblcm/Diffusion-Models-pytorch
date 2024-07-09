import torch
from ddpm import Diffusion
from modules.motion.dataset import get_pair_loader
import argparse
import os
import numpy
from modules.motion.tinynet_20240618_vanilla import TinyNet
import torch.nn as nn
from tqdm import tqdm
from utils.shared import *

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1
args.shape_one = (64, 19, 3)
args.path_dataset = os.path.join("data", "temp.npy")
args.device = "cuda"

def get_path_last_model(path_dir):
    list_name_models = os.listdir(path_dir)
    list_name_models = sorted(list(filter(lambda name: len(name) > 3 and name[-3:] == '.pt', list_name_models)))
    return os.path.join(path_dir, list_name_models[-1])

diff = Diffusion(args.shape_one, device=args.device)
model = TinyNet()
run_name = model.run_name
path_dir_output = get_path_dir_output(run_name)
model.load_state_dict(torch.load(get_path_last_model(os.path.join(path_dir_output, "models"))))
model = model.to(args.device)
model.eval()

# def test_prediction(list_step):
#     count_step = len(list_step)
#     mse = nn.MSELoss()
#     image = next(iter(dataloader))[0].to(args.device)
#     t = torch.Tensor(list_step).long().to(args.device)
#     image = image[None,:,:,:].repeat(count_step, 1, 1, 1)
#     noised_image, noise = diff.noise_images(image, t)
#     predicted_noise = model(noised_image, t)

#     list_loss = []
#     for index_step in range(count_step):
#         list_loss.append([list_step[index_step], mse(noise[index_step], predicted_noise[index_step]).detach().cpu().numpy()])
#     list_loss = numpy.array(list_loss)
#     print(list_loss)

#     print(nn.MSELoss()(noise, predicted_noise).detach().cpu().numpy())
# test_prediction([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 999])

# data: (step,frame,joint,dim)
def save_frame_step(path_root, data):
    path_dir = os.path.join(path_root, "frames")
    os.makedirs(path_dir, exist_ok=True)
    for step in range(data.shape[0]):
        image = data[step,:,:,:]
        numpy.save(os.path.join(path_dir, "{0:05d}.npy".format(data.shape[0] - step - 1)), image)

    count_frame = data.shape[1]
    path_dir = os.path.join(path_root, "steps")
    os.makedirs(path_dir, exist_ok=True)
    for index_frame in range(count_frame):
        image = data[:,index_frame,:,:]
        numpy.save(os.path.join(path_dir, "{0:05d}.npy".format(index_frame)), image)

# x_noised: (batch,frame,joint,dim)
def recover_and_save(path_dir_output_test, x_noised):
    x = x_noised
    xs = [x.cpu().numpy()]
    for i in tqdm(reversed(range(1, diff.noise_steps + 1)), position=0):
        x = diff.denoise_images_single_step(model, x, i)
        xs.append(x.cpu().numpy())
    xs = numpy.array(xs) # shape:(time+1,batch,frame,joint,dim)
    
    save_frame_step(path_dir_output_test, xs[:,0,:,:,:])
    numpy.save(os.path.join(path_dir_output_test, "noised.npy"), x_noised[0].cpu().numpy())

def test_recover_from_image():
    path_dir_output_test = os.path.join(path_dir_output, "test_recover_image")
    os.makedirs(path_dir_output_test, exist_ok=True)

    dataloader, _ = get_pair_loader(args)
    image = next(iter(dataloader)).to(args.device)
    numpy.save(os.path.join(path_dir_output_test, "original.npy"), image[0].cpu().numpy())
    
    x_noised, _ = diff.noise_images(image, torch.Tensor([diff.noise_steps - 1] * image.shape[0]).long().to(args.device)) # shape:(batch,frame,joint,dim)
    recover_and_save(path_dir_output_test, x_noised)

def test_recover_from_noise():
    path_dir_output_test = os.path.join(path_dir_output, "test_recover_noise")
    os.makedirs(path_dir_output_test, exist_ok=True)

    x_noised = torch.randn([args.batch_size] + list(args.shape_one)).to(args.device)
    recover_and_save(path_dir_output_test, x_noised)

def test_sampling():
    path_dir_output_test = os.path.join(path_dir_output, "test_sampling")
    os.makedirs(path_dir_output_test, exist_ok=True)

    xs = diff.sample_save_steps(model, 1) # shape:(time+1,batch,frame,joint,dim)
    save_frame_step(path_dir_output_test, xs[:,0,:,:,:].cpu().numpy())

test_recover_from_image()
test_recover_from_noise()
test_sampling()