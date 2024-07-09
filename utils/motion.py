import os
from .shared import get_path_dir_output
import numpy
import torch

def sample_and_save(run_name, epoch, diffusion, model):
    path_dir_output = get_path_dir_output(run_name)

    sampled_images = diffusion.sample_save_steps(model, n=1)
    sampled_images = torch.swapaxes(sampled_images, 0, 1)[0] # Get single item in batch, (step,frame,joint,dim)
    sampled_images_step = torch.swapaxes(sampled_images, 0, 1)[0].to('cpu').numpy() # Get single frame, (step,joint,dim)
    sampled_images_frame = sampled_images[0].to('cpu').numpy() # Get single step multi frame, (frame,joint,dim)
    # sampled_images = dataset.add_bias(sampled_images)

    path_dir_steps = os.path.join(path_dir_output, "result_steps")
    path_dir_frames = os.path.join(path_dir_output, "result_frames")
    os.makedirs(path_dir_steps, exist_ok=True)
    os.makedirs(path_dir_frames, exist_ok=True)

    numpy.save(os.path.join(path_dir_steps, "epoch_{0:05d}.npy".format(epoch)), sampled_images_step)
    numpy.save(os.path.join(path_dir_frames, "epoch_{0:05d}.npy".format(epoch)), sampled_images_frame)