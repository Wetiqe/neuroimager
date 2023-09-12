import os
import tqdm
import warnings
import pandas as pd
import nibabel as nib

# def _copy_image(image):
#     image_data_copy = image.get_fdata().copy()
#     affine_copy = image.affine.copy()
#     return nib.Nifti1Image(image_data_copy, affine_copy)


def install_workbench(dir):
    os.environ["PATH"] += ":/content/workbench/bin_rh_linux64"
    # https://humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
    # https://humanconnectome.org/storage/app/media/workbench/workbench-mac64-v1.5.0.zip
    # https://humanconnectome.org/storage/app/media/workbench/workbench-windows64-v1.5.0.zip
    # !wget --no-check-certificate https://humanconnectome.org/storage/app/media/workbench/workbench-rh_linux64-v1.5.0.zip
    # !unzip -q workbench-rh_linux64-v1.5.0.zip -d /content
    # !pip install neuromaps tqdm --quiet


def compare_all_neuromaps(
    user_map: str,
    src_space="MNI152",
    MNI_trg_space=("fsaverage", "10k"),
    skip_source=None,
    return_nulls=True,
    method="linear",
):
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
            raise ValueError(f"space {space} not implemented",)

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
