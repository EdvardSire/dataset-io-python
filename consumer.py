from pathlib import Path
import struct
import cv2
from shm_dataset_io import ShmDatasetConsumer


if __name__ == "__main__":
    sdc = ShmDatasetConsumer(Path("/home/user/datasets/UAV_VisLoc_dataset/04/drone"))
    ms, n, h, w, c = struct.unpack_from("5i", sdc._shm.buf)
    print(f"{n} images of: {h}x{w}x{c}")
    cv2.imshow("", sdc.data[0]); cv2.waitKey(0)
    sdc.cleanup()
