# robust loading functions
import csv
import pandas as pd
import nibabel as nib
import bids


def rbload_csv(csv_file):
    if isinstance(csv_file, str):
        with open(csv_file, "r") as file:
            dialect = csv.Sniffer().sniff(file.read(1024))
            file.seek(0)
            return pd.read_csv(csv_file, delimiter=dialect.delimiter)
    elif isinstance(csv_file, pd.DataFrame):
        return csv_file


def rbload_imgs(
    imgs: list or str or nib.nifti1.Nifti1Image or bids.layout.models.BIDSImageFile,
):
    """
    Load images from a list of paths, a single path, or a list of nibabel images
    Return a list of nibabel images
    """
    if isinstance(imgs, str):
        return [nib.load(imgs)]
    elif isinstance(imgs, nib.nifti1.Nifti1Image):
        return [imgs]
    elif isinstance(imgs, bids.layout.models.BIDSImageFile):
        return imgs.get_image()
    try:
        imgs = list(imgs)
    except TypeError:
        raise ValueError(
            "If imgs is not a single file, then imgs must can be converted to list"
        )
    try:
        if isinstance(imgs[0], str):
            imgs = [nib.load(stat_map) for stat_map in imgs]
        elif isinstance(imgs[0], bids.layout.models.BIDSImageFile):
            imgs = [img.get_image() for img in imgs]
        elif isinstance(imgs[0], nib.nifti1.Nifti1Image):
            pass
        else:
            raise ValueError("imgs must be list of str/nib.Nifti1Image/BIDSImageFile")
    except IndexError:
        print("imgs must be non-empty")

    return imgs
