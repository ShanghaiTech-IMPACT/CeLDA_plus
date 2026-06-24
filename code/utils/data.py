from torch.utils.data import DataLoader

from dataloaders.landmark_dataset import CL_Landmark_Mix
from .helpers import get_augmentations


def build_mix_dataloaders(
    data_path,
    image_size,
    batch_size,
    num_workers,
    rotation_limit=10,
    use_elastic=True,
    use_flip=False,
):
    aug = get_augmentations(
        image_size=tuple(image_size),
        rotation_limit=rotation_limit,
        use_elastic=bool(use_elastic),
        use_flip=bool(use_flip),
    )
    train_data = CL_Landmark_Mix(base_dir=data_path, splits="train", augmentation=aug["train"])
    val_data = CL_Landmark_Mix(base_dir=data_path, splits="val", augmentation=aug["val"])
    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    valloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return trainloader, valloader
