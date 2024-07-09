import os
import torch
import numpy

def check_finite(xs):
    xs = torch.isfinite(xs)
    xs = torch.flatten(xs)
    xs = xs.cpu().numpy()
    for x in xs:
        if not x:
            return False
    return True

def get_path_dir_output(run_name):
    return os.path.join("output", run_name)

# Read latest trained model and epoch. If no model exists, 0 is returned.
def load_model(run_name, model):
    path_dir_output = get_path_dir_output(run_name)
    path_dir_model = os.path.join(path_dir_output, "models")
    epoch = 0

    if os.path.isdir(path_dir_model):
        list_name_file = sorted(os.listdir(path_dir_model))
        list_name_file.reverse()
        i = 0
        while epoch <= 0 and i < len(list_name_file):
            name_file = list_name_file[i]
            if len(name_file) > 3 and name_file[-3:] == ".pt":
                s = name_file[:-3]
                if s.isdigit():
                    epoch = int(s)
                    path_file = os.path.join(path_dir_model, name_file)
                    model.load_state_dict(torch.load(path_file))
                    print("Loaded {}".format(path_file))
            i = i + 1
    return epoch

def save_model(run_name, model, epoch):
    path_dir_output = get_path_dir_output(run_name)
    path_dir_model = os.path.join(path_dir_output, "models")
    os.makedirs(path_dir_model, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path_dir_model, "{0:05d}.pt".format(epoch)))

def save_error(
    run_name,
    epoch,
    output_tlist_item_count,
    output_tlist_mse_sum,
    train_time
):
    path_dir_output = get_path_dir_output(run_name)
    path_dir_error = os.path.join(path_dir_output, "errors")
    os.makedirs(path_dir_error, exist_ok=True)

    numpy.savez(
        os.path.join(path_dir_error, "{0:05d}.npz".format(epoch)),
        tlist_count=output_tlist_item_count,
        tlist_mse_sum=output_tlist_mse_sum,
        total_count=numpy.sum(output_tlist_item_count),
        total_mse_sum=numpy.sum(output_tlist_mse_sum),
        train_time=train_time
    )

class Dumper:
    def __init__(self, path_dir, name_format):
        self.save_index = 0
        self.real_index = 0
        self.real_count = 1
        self.path_dir = path_dir
        self.name_format = name_format

    def dump(self, data, batched):
        os.makedirs(self.path_dir, exist_ok=True)
        if batched:
            for index_batch in range(data.shape[0]):
                self.dump_single(data[index_batch])
        else:
            self.dump_single(data)

    def dump_single(self, data):
        if self.real_index >= self.real_count:
            assert len(data.shape) == 3
            numpy.save(os.path.join(self.path_dir, self.name_format.format(self.save_index)), data.cpu().numpy())

            self.real_count = self.real_count * 2
            self.save_index = self.save_index + 1
        self.real_index = self.real_index + 1