from pathlib import Path
from dataset_io import ShmDataset


if __name__ == "__main__":
    ShmDataset.load_from_path(Path("/home/user/datasets/UAV_VisLoc_dataset/04/drone"))
