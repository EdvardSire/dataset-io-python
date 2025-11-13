from pathlib import Path
import cv2
from dataset_io import ShmDatasetConsumer


if __name__ == "__main__":
    sdc = ShmDatasetConsumer(Path("/home/user/datasets/UAV_VisLoc_dataset/04/drone"))
    for i in range(10):
        cv2.imshow("", sdc.data[i])
        cv2.waitKey(0)
    sdc.cleanup()
