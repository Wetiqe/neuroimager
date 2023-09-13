import os
import tqdm
import warnings
import pandas as pd
import nibabel as nib
import warnings

# def _copy_image(image):
#     image_data_copy = image.get_fdata().copy()
#     affine_copy = image.affine.copy()
#     return nib.Nifti1Image(image_data_copy, affine_copy)


def install_workbench(path, version="1.5.0", redhat: bool = "warn"):
    import platform
    import zipfile
    import urllib.request
    from pathlib import Path

    workbench_urls = {
        "Linux": f"https://humanconnectome.org/storage/app/media/workbench/workbench-linux64-v{version}.zip",
        "RedHat": f"https://humanconnectome.org/storage/app/media/workbench/workbench-rh_linux64-v{version}.zip",
        "Darwin": f"https://humanconnectome.org/storage/app/media/workbench/workbench-mac64-v{version}.zip",
        "Windows": f"https://humanconnectome.org/storage/app/media/workbench/workbench-windows64-v{version}.zip",
    }

    os_type = platform.system()
    if os_type == "Linux":
        distribution = platform.freedesktop_os_release()["ID"]
        if distribution in ["rhel", "centos", "fedora", "red hat"]:
            os_type = "RedHat"
        elif distribution in ["ubuntu", " debian", "arch", "opensuse", "sles"]:
            os_type = "Linux"
        else:
            if redhat == "warn":
                warnings.warn(
                    "Your distribution is not support by this function, but you can choose whether this is a RedHat machine or not"
                )
                return
            elif redhat is True:
                os_type = "RedHat"
            elif redhat is False:
                os_type = "Linux"

    workbench_url = workbench_urls.get(os_type, None)

    if workbench_url is None:
        raise ValueError(f"Unsupported operating system: {os_type}")

    workbench_zip = f"workbench-{os_type.lower()}-v{version}.zip"
    workbench_dir = Path(path)

    print(f"Downloading {workbench_zip}...")
    urllib.request.urlretrieve(workbench_url, workbench_zip)

    print(f"Extracting {workbench_zip} to {workbench_dir}...")
    with zipfile.ZipFile(workbench_zip, "r") as zip_ref:
        zip_ref.extractall(workbench_dir)

    bin_paths = {
        "Linux": "bin_linux64",
        "RedHat": "bin_rh_linux64",
        "Darwin": "bin_darwin64",
        "Windows": "bin_windows64",
    }
    os.environ["PATH"] += f":{path}/workbench/" + bin_paths[os_type]


def compare_all_neuromaps(
    user_map: str,
    src_space="MNI152",
    MNI_trg_space=("fsaverage", "10k"),
    skip_source=None,
    return_nulls=True,
    method="linear",
):
    """
    This function compares a user map to all available neuromaps returned by `neuromaps.datasets.available_annotations()`
    If the neuromap is in surface

    """
    # TODO: change to make use of additional requirements
    try:
        import neuromaps
    except ImportError:
        raise ImportError("neuromaps package is not found. Try `pip install neuromaps`")
    from neuromaps import transforms, stats, images, nulls
    from neuromaps.datasets import available_annotations, fetch_annotation
    from neuromaps.resampling import resample_images

    if skip_source is None:
        skip_source = []
    result = []
    for annotation in tqdm.tqdm(available_annotations()):
        state_map = user_map  # use rbloads()
        source, map_name, space, resolution = annotation
        annotation = list(annotation)
        if source in skip_source:
            continue
        neuro_map = fetch_annotation(source=source, desc=map_name)
        if space == "MNI152":
            space, resolution = MNI_trg_space
            state_map, neuro_map = resample_images(
                state_map,
                neuro_map,
                src_space=src_space,
                trg_space=space,
                method=method,
                resampling="transform_to_alt",
                alt_spec=(space, resolution),
            )

        elif space == "fsaverage":
            state_map = transforms.mni152_to_fsaverage(state_map, resolution)
        elif space == "fsLR":
            if resolution in ["4k"]:
                warnings.warn("resampling to fsLR 4k is not supported, setting to 32k")
                resolution = "32k"
                state_map, neuro_map = resample_images(
                    state_map,
                    neuro_map,
                    src_space=src_space,
                    trg_space=space,
                    method=method,
                    resampling="transform_to_alt",
                    alt_spec=(space, resolution),
                )
            else:
                state_map = transforms.mni152_to_fslr(state_map, resolution)

        elif space == "civet":
            state_map = transforms.mni152_to_civet(state_map, civet_density=resolution)
        else:
            raise ValueError(
                f"space {space} not implemented",
            )

        rotated = nulls.alexander_bloch(
            neuro_map, atlas=space, density=resolution, n_perm=100, seed=1234
        )
        if return_nulls:
            corr, pval, nulls = stats.compare_images(
                state_map, neuro_map, nulls=rotated, return_nulls=return_nulls
            )
            annotation.extend([corr, pval, nulls])
            cols = ["source", "map", "space", "resolution", "corr", "pval", "null"]
        else:
            corr, pval = stats.compare_images(
                state_map, neuro_map, nulls=rotated, return_nulls=return_nulls
            )
            annotation.extend([corr, pval])
            cols = ["source", "map", "space", "resolution", "corr", "pval"]
        result.append(annotation)
    result_df = pd.DataFrame(result)
    result_df.columns = cols
    return result_df


def pretrained_neurosynth_model(method="correlation", save_path="./"):
    import gzip
    import pickle
    import requests
    import os

    # Download and load the decoder
    url = f"https://raw.githubusercontent.com/wetiqe/neuroimager/main/asset/decoder/neurosynth/{method}_decoder.pkl.gz"
    response = requests.get(url)

    if response.status_code == 200:
        if save_path:
            decoder_path = os.path.join(save_path, f"{method}_decoder.pkl.gz")
            with open(decoder_path, "wb") as f:
                f.write(response.content)
            with gzip.open(decoder_path, "rb") as f:
                decoder = pickle.load(f)
        else:
            with gzip.open(response.content, "rb") as f:
                decoder = pickle.load(f)
    else:
        print(
            f"Error: Unable to download the file (status code {response.status_code})"
        )

    # Download and load the features
    url = f"https://raw.githubusercontent.com/wetiqe/neuroimager/main/asset/decoder/neurosynth/selected_features.txt"
    response = requests.get(url)

    if response.status_code == 200:
        if save_path:
            features_path = os.path.join(save_path, "selected_features.txt")
            with open(features_path, "wb") as f:
                f.write(response.content)
            with open(features_path, "r") as f:
                features = f.read().splitlines()
        else:
            features = response.text.splitlines()
    else:
        print(
            f"Error: Unable to download the file (status code {response.status_code})"
        )

    return decoder, features
