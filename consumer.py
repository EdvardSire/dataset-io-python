from pathlib import Path
import numpy as np
import cv2
from compat import SharedMemory

shm_name = "f280135cd2844fd4b463aec7cd49f00f"
existing_shm = SharedMemory(name=shm_name, create=False, track=False)

image_paths = sorted(Path("/home/user/datasets/UAV_VisLoc_dataset/04/drone").iterdir())
image = cv2.imread(image_paths[0].__str__())
h, w, c = image.shape
n = len(image_paths)
data = np.frombuffer(existing_shm.buf, dtype=np.uint8).reshape((n, h, w, c))
print(data[0].shape)
cv2.imshow("", data[0])
cv2.waitKey(0)

del data
existing_shm.close()

