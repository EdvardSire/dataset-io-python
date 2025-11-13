from pathlib import Path
from compat import SharedMemory
from hashlib import sha256
import struct

import numpy as np
import cv2

class ShmDatasetConsumer():
    def __init__(self, image_dir: Path):
        shm_name = sha256(image_dir.home().__str__().encode()).hexdigest()
        self.shm = SharedMemory(name=shm_name, create=False, track=False)
        ms, n, h, w, c = struct.unpack_from('5i', self.shm.buf, offset=0)
        self.data =  np.frombuffer(self.shm.buf, dtype=np.uint8, offset=ms).reshape((n, h, w, c))

    def cleanup(self):
        del self.data
        self.shm.close()


if __name__ == "__main__":
    sdc = ShmDatasetConsumer(Path("/home/user/datasets/UAV_VisLoc_dataset/04/drone"))

    for i in range(len(sdc.data)):
        cv2.imshow("", sdc.data[i])
        cv2.waitKey(0)

    sdc.cleanup()
