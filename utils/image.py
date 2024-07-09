import os
from .shared import get_path_dir_output
import numpy
import torch
import torchvision
from PIL import Image

def save_images(images, path, **kwargs):
    images = images.permute(0, 3, 1, 2) # BHWC->BCHW
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def sample_and_save(run_name, epoch, diffusion, model, grid=4):
    path_dir_output = get_path_dir_output(run_name)

    sampled_images = diffusion.sample_save_steps(model, grid) # (time+1,batch,x,y,chan)
    sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2
    sampled_images = (sampled_images * 255).type(torch.uint8)

    sampled_images_step = sampled_images[:,0,...] # Get single image, multi step. (time+1,x,y,chan)
    sampled_images_batch = sampled_images[0] # Get single step, multi image. (batch,x,y,chan)

    # change 1000 steps to SIZE_BATCH steps.
    indexes = numpy.linspace(0, sampled_images.shape[0] - 1, sampled_images.shape[1]).astype(int)
    sampled_images_step = sampled_images_step[indexes]

    path_dir_steps = os.path.join(path_dir_output, "result_steps")
    path_dir_batches = os.path.join(path_dir_output, "result_batches")
    os.makedirs(path_dir_steps, exist_ok=True)
    os.makedirs(path_dir_batches, exist_ok=True)

    save_images(sampled_images_step, os.path.join(path_dir_steps, "epoch_{0:05d}.png".format(epoch)))
    save_images(sampled_images_batch, os.path.join(path_dir_batches, "epoch_{0:05d}.png".format(epoch)))