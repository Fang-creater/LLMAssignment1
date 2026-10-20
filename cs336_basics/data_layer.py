import numpy as np

def create_dataloader(data_path: str) -> np.ndarray:
    data = np.load(data_path, mmap_mode="r")
    print(f"[Data] Loaded {data_path}: shape={data.shape}, dtype={data.dtype}")
    assert data.dtype in (np.int16, np.int32, np.int64, np.uint16), \
        "Expected int dtype, got {}".format(data.dtype)
    return data