
import os
from glob import glob

#from tqdm import tqdm

from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    ResizeWithPadOrCropd,

)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism,first



#in_dir = r'C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\atlas-train-dataset-1.0.1\train'
in_dir = r"C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\atlas-train-dataset-1.0.1\train"
#in_dir = r"../atlas-train-dataset-1.0.1/atlas-train-dataset-1.0.1/train"


def prepare(in_dir, pixdim=(1.5, 1.5, 1.0),  spatial_size=[448, 352, 64], cache=True, batch_size=2):
    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))
    #print(len(path_train_volumes))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-50, a_max=1250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            #Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ResizeWithPadOrCropd(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-50, a_max=1250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            #Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ResizeWithPadOrCropd(keys=["vol", "seg"], spatial_size=(448, 352, 64)),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=0.5)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        return train_loader, test_loader