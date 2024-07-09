import os
import sys

def process(path_dir):
    list_path_file = []
    for name_file in os.listdir(path_dir):
        path_file = os.path.join(path_dir, name_file)
        if os.path.isdir(path_file):
            process(path_file)
        if os.path.isfile(path_file):
            list_path_file.append(path_file)

    for path_file in list_path_file:
        if not ((len(path_file) >= 3) and (path_file[-3:] == ".pt")):
            return
    if len(list_path_file) < 1:
        return
    
    list_path_file = sorted(list_path_file)
    for path_file in list_path_file[:-1]:
        os.remove(path_file)

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) < 2:
        print("python clear_models.py PATH_DIR")
    else:
        for arg in sys.argv[1:]:
            process(arg)