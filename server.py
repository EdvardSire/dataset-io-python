from pathlib import Path
import gc
import uuid
import os

import numpy as np
import cv2

from multiprocessing import shared_memory


b2gb = 1024**3



def shm_limit_and_used_bytes(path="/dev/shm"):
    # findmnt -o AVAIL,USED /dev/shm
    # sudo mount -o remount,size=XXG /dev/shm
    st = os.statvfs(path)
    limit = st.f_bavail * st.f_frsize  # available bytes to non-root
    used  = (st.f_blocks - st.f_bfree) * st.f_frsize  # used bytes
    return limit, used


def available_system_memory_bytes(meminfo = Path("/proc/meminfo")) -> float:
    with open(meminfo) as f:
        meminfo = {line.split(':')[0]: int(line.split()[1]) for line in f}
        return meminfo["MemAvailable"] * 1024


def profile_dataset(image_paths: list[Path]):
    i1 = cv2.imread(image_paths[0].__str__())
    h, w, c = i1.shape
    datatype = i1.dtype
    n = len(image_paths)

    return n, h, w, c, datatype


def assert_feasible_allocation(needed_bytes):
    avail_system_bytes = available_system_memory_bytes()
    shm_limit_bytes, _ = shm_limit_and_used_bytes()
    pressure = needed_bytes/avail_system_bytes

    if needed_bytes > shm_limit_bytes:
        raise RuntimeError(f"Increase /dev/shm size")

    if pressure > 0.9:
        raise RuntimeError(f"""Not comfortable allocating:
                                   pressure     {pressure:.2f}
                                   needed/avail/shm-limit {needed_bytes/b2gb:.2f}/{avail_system_bytes/b2gb:.2f}/{shm_limit_bytes/b2gb:.2f} GB""")
    print(f"Memory pressure: {pressure:.2f}")
    return


def allocate_shared_slab(slab_size, name = uuid.uuid4().hex):
    shm = shared_memory.SharedMemory(create=True, size=slab_size, name=name)
    return shm, name


if __name__ == "__main__":
    image_paths = sorted(Path("/home/user/datasets/UAV_VisLoc_dataset/04/drone").iterdir())
    n, h, w, c, datatype = profile_dataset(image_paths)
    slab_size_bytes = n*h*w*c*np.dtype(datatype).itemsize
    assert_feasible_allocation(slab_size_bytes)
    shm, name = allocate_shared_slab(slab_size_bytes)
    arr = np.ndarray((n, h, w, c), dtype=datatype, buffer=shm.buf)
    gc.collect()

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path.__str__())
        if img.shape != (h, w, c):
            raise ValueError(f"Image {image_path} has unexpected shape {img.shape}")
        arr[i] = img  

    try:
        print(f"Spinning with allocated slab: {name} ; size {slab_size_bytes/b2gb:.2f} GB")
        while True:
            pass
    except KeyboardInterrupt:
        shm.close()
        shm.unlink()
