import errno
import os


def ensure_files_exist():
    dirpath = os.path.abspath("data")

    if not os.path.exists(dirpath):
        raise FileNotFoundError(errno.ENOENT)
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(errno.ENOTDIR)
    if not os.access(dirpath, os.R_OK):
        raise FileNotFoundError(errno.EACCES)
    for file in ["t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "train-images.idx3-ubyte", "train-labels.idx1-ubyte"]:
        filepath = os.path.join(dirpath, file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(errno.ENOENT)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(errno.EISDIR)
        if not os.access(filepath, os.R_OK):
            raise FileNotFoundError(errno.EACCES)

def main():
    try:
        ensure_files_exist()
    except FileNotFoundError:
        print(f"All four files from the original mnist dataset are required to exists in the directory /data.")
        exit(1)




if __name__ == "__main__":
    main()