import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy
from utils.shared import *
# from utils.motion import *
# from modules.motion.tinynet_20240618_vanilla import TinyNet
# from modules.motion.dataset import get_pair_loader
from utils.image import *
from modules.image.vanilla import UNet
from modules.image.dataset import get_pair_loader
import time
import math

class Diffusion:
    # Generate array of alpha, ahpla_hat. They have size of count_step + 1.
    # If count_step_fix is True, count_step_fix
    def prepare_noise_schedule(grow_h, count_step_max, alpha_hat_st, alpha_hat_ed):
        # float32 has accuracy of 6.9236899 digits. log_10(2^23)=6.9236899
        # thus if alpha[1]=alpha_hat[1] is very close to 1, data will be invalid.
        order_max_max = math.log(alpha_hat_ed, alpha_hat_st)
        binomial_curr = numpy.array([1] + [0] * grow_h)
        for _ in range(grow_h):
            binomial_curr = binomial_curr + numpy.concatenate(([0], binomial_curr[:-1]))
        list_order_h = [0]
        while (len(list_order_h) <= count_step_max) and (binomial_curr[-1] < order_max_max):
            list_order_h.append(binomial_curr[-1])
            binomial_curr = binomial_curr + numpy.concatenate(([0], binomial_curr[:-1]))
        list_order_a = list_order_h - numpy.concatenate(([0], list_order_h[:-1]))
        list_order_a = torch.tensor(list_order_a)
        list_order_h = torch.tensor(list_order_h)
        order_max = list_order_h[-1].item()
        return torch.pow(alpha_hat_ed, list_order_a / order_max), torch.pow(alpha_hat_ed, list_order_h / order_max)

    def __init__(self, shape_one, count_step_max=1000, alpha_hat_st=0.9999, alpha_hat_ed=4e-5, alpha_hat_grow=2, device="cuda"):
        self.shape_one = shape_one
        self.device = device

        self.alpha, self.alpha_hat = Diffusion.prepare_noise_schedule(alpha_hat_grow, count_step_max, alpha_hat_st, alpha_hat_ed)
        self.noise_steps = self.alpha.shape[0] - 1
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        self.beta = 1. - self.alpha

    # Given x_0 and t, return x_t.
    def noise_images(self, x_0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        assert check_finite(sqrt_alpha_hat)
        assert check_finite(sqrt_one_minus_alpha_hat)
        Ɛ = torch.randn_like(x_0)
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    # Returns an int in [1,self.noise_steps]
    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,)) + 1

    # n: batch size
    # Return shape: (batch,frame,joint,dim)
    def sample(self, model, n):
        print("Sampling {} new images....".format(n))
        shape_mul = [n] + list(self.shape_one)
        model.eval()
        with torch.no_grad():
            x = torch.randn(shape_mul).to(self.device)
            for step in tqdm(reversed(range(1, self.noise_steps + 1)), position=0):
                x = self.internal_denoise_single_step(model, x, step)
                assert check_finite(x)
        model.train()
        return x

    # n: batch size
    # Return shape: (time+1,batch,frame,joint,dim)
    def sample_save_steps(self, model, n):
        xs = []
        print("Sampling {} new images....".format(n))
        shape_mul = [n] + list(self.shape_one)
        model.eval()
        with torch.no_grad():
            x = torch.randn(shape_mul).to(self.device)
            xs.append(x)
            for step in tqdm(reversed(range(1, self.noise_steps + 1)), position=0):
                x = self.internal_denoise_single_step(model, x, step)
                assert check_finite(x)
                xs.append(x)
        model.train()
        return torch.stack(list(reversed(xs)))

    # x: (batch,frame,joint,dim)
    # Reduce x_t to x_t-1
    def denoise_images_single_step(self, model, x, step):
        model.eval()
        with torch.no_grad():
            x = self.internal_denoise_single_step(model, x, step)
        model.train()
        return x
    
    # x: (batch,frame,joint,dim)
    # Reduce x_t to x_t-1
    def internal_denoise_single_step(self, model, x, step):
        t = (torch.ones(x.shape[0]) * step).long().to(self.device)
        predicted_noise = model(x, t)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        if step > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        return 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

def train(args):
    device = args.device
    model = UNet().to(device)
    run_name = model.run_name
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse_func = nn.MSELoss(reduction='none')
    diffusion = Diffusion(
        args.shape_one,
        device=device,
        alpha_hat_grow=args.alpha_hat_grow,
        alpha_hat_st=args.alpha_hat_st,
        alpha_hat_ed=args.alpha_hat_ed,
        count_step_max=args.count_step_max
    )
    dataloader, testloader = get_pair_loader(args)
    if ("name_st" in args) and (not (args.name_st is None)) and (len(args.name_st) > 0):
        run_name = "{}_{}".format(args.name_st, run_name)
    if ("name_ed" in args) and (not (args.name_ed is None)) and (len(args.name_ed) > 0):
        run_name = "{}_{}".format(run_name, args.name_ed)
    path_dir_output = get_path_dir_output(run_name)

    dumper = Dumper(os.path.join(path_dir_output, "dumps"), "{0:05d}.npy")

    train_time = 0 # used to record time consumed in a single train epoch.

    epoch = load_model(run_name, model)
    while True:
        if epoch % args.epoch_save_image == 0: # Sample Images.
            sample_and_save(run_name, epoch, diffusion, model)

        if True: # run test.
            print("Testing epoch {}:".format(epoch))
            pbar = tqdm(testloader)

            ts = torch.linspace(1, diffusion.noise_steps + 1, testloader.count_item).long().clamp(max=diffusion.noise_steps).to(device)
            index_item = 0
            # count_batch = len(testloader)

            output_tlist_item_count = numpy.zeros(diffusion.noise_steps + 1, dtype=numpy.dtype(int))
            output_tlist_mse_sum = numpy.zeros(diffusion.noise_steps + 1)
            mse_sum_total = 0

            model.eval()
            with torch.no_grad():
                for index_batch, (x_0, _) in enumerate(pbar):
                    x_0 = x_0.to(device)

                    size_batch = x_0.shape[0]

                    t = ts[index_item:index_item + size_batch]
                    x_t, noise = diffusion.noise_images(x_0, t)
                    predicted_noise = model(x_t, t)

                    mse = mse_func(noise, predicted_noise).mean(dim=(1, 2, 3)) # shape: (batch,frame,joint,dim)

                    mse_sum_batch = mse.sum().item()
                    mse_sum_total = mse_sum_total + mse_sum_batch
                    mse_ave_batch = mse_sum_batch / size_batch
                    mse_ave_total = mse_sum_total / (index_item + size_batch)

                    # Save data.
                    t = t.to('cpu').numpy()
                    mse = mse.to('cpu').numpy()
                    for i in range(size_batch):
                        t_single = t[i]
                        output_tlist_item_count[t_single] = output_tlist_item_count[t_single] + 1
                        output_tlist_mse_sum[t_single] = output_tlist_mse_sum[t_single] + mse[i]

                    t_single = t[-1]
                    pbar.set_postfix(mse_batch=mse_ave_batch, mse_total=mse_ave_total, T=t_single)
                    index_item = index_item + size_batch
            torch.cuda.empty_cache()
            model.train()

            save_error(run_name, epoch, output_tlist_item_count, output_tlist_mse_sum, train_time)

        if epoch % args.epoch_save_model == 0:
            save_model(run_name, model, epoch)
        
        epoch = epoch + 1

        if True: # run train.
            print("Training epoch {}:".format(epoch))
            pbar = tqdm(dataloader)
            train_time = 0
            mse_sum_total = 0
            index_item = 0
            for index_batch, (x_0, _) in enumerate(pbar):
                dumper.dump(x_0, True)

                train_time_prev = time.time()
                x_0 = x_0.to(device)
                t = diffusion.sample_timesteps(x_0.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(x_0, t)
                predicted_noise = model(x_t, t)

                mse_sum_batch = mse_func(noise, predicted_noise).mean(dim=(1, 2, 3)).sum()

                optimizer.zero_grad()
                mse_sum_batch.backward()
                optimizer.step()

                mse_sum_batch = mse_sum_batch.item()
                mse_sum_total = mse_sum_total + mse_sum_batch
                mse_ave_batch = mse_sum_batch / size_batch
                mse_ave_total = mse_sum_total / (index_item + size_batch)
                index_item = index_item + size_batch
                
                train_time = train_time + time.time() - train_time_prev
                pbar.set_postfix(mse_batch=mse_ave_batch, mse_total=mse_ave_total)
            torch.cuda.empty_cache()

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-csm", "--countstepmax", default=1000, type=int, dest='count_step_max')
    parser.add_argument("-ahg", "--alphahatgrow", default=1, type=int, dest='alpha_hat_grow')
    parser.add_argument("-ahs", "--alphahatst", default=0.9999, type=float, dest='alpha_hat_st')
    parser.add_argument("-ahe", "--alphahated", default=4e-5, type=float, dest='alpha_hat_ed')
    parser.add_argument("-nst", "--namestart", default="", type=str, dest='name_st')
    parser.add_argument("-ned", "--nameend", default="", type=str, dest='name_ed')

    args = parser.parse_args()
    # args.batch_size = 128
    # args.shape_one = (64, 19, 3)
    # # args.path_dataset = os.path.join("data", "motion", "temp.npy")
    # args.path_dataset = os.path.join("data", "motion", "dataset_train.npz")
    # args.path_testset = os.path.join("data", "motion", "dataset_test.npz")

    args.batch_size = 4
    args.shape_one = (64, 64, 3)
    args.path_dataset = os.path.join("data", "image", "StanfordCars", "cars_train")
    args.path_testset = os.path.join("data", "image", "StanfordCars", "cars_test")

    args.device = "cuda"
    args.lr = 3e-4
    args.epoch_save_model = 1
    args.epoch_save_image = 1
    train(args)

if __name__ == '__main__':
    launch()
