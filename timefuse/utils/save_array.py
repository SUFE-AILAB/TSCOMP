import numpy as np
import h5py
import os
import time


def save_arr(arr, file_path, file_type="h5", create_dir=True, dtype=None, verbose=True):
    start_time = time.time()

    if dtype is not None:
        arr = arr.astype(dtype)

    if create_dir:
        if not os.path.exists(file_path) and verbose:
            print(f"Creating directory: {os.path.dirname(file_path)}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file_type == "h5":
        with h5py.File(file_path, "w") as f:
            f.create_dataset("arr", data=arr, compression="gzip")
    elif file_type == "npy":
        np.save(file_path, arr)
    else:
        raise ValueError("Unsupported file_type. Use 'h5' or 'npy'.")

    if verbose:
        end_time = time.time()
        file_size = os.path.getsize(file_path) / (1024**2)
        print(
            f"Array has been saved to '{file_path}', size: {file_size:.2f} MB ({end_time - start_time:.2f}s)"
        )


def load_arr(file_path, file_type="h5", verbose=True):
    start_time = time.time()

    if file_type == "h5":
        with h5py.File(file_path, "r") as f:
            arr = f["arr"][:]
    elif file_type == "npy":
        arr = np.load(file_path, allow_pickle=True)
    else:
        raise ValueError("Unsupported file_type. Use 'h5' or 'npy'.")

    if verbose:
        end_time = time.time()
        print(
            f"Array has been loaded from '{file_path}' ({end_time - start_time:.2f}s)"
        )

    return arr
