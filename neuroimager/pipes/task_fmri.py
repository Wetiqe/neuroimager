import csv
import os
from tqdm import tqdm
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import nibabel as nib
import bids.layout.models

from nilearn.glm.first_level import FirstLevelModel, check_design_matrix
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.image import clean_img
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.plotting import plot_stat_map, view_img_on_surf, plot_glass_brain


class TaskFmri(BaseEstimator, TransformerMixin, object):
    def __init__(self, tr: int, contrasts: list or dict, out_dir: str):
        self.tr = tr
        self.contrasts = contrasts
        self.out_dir = out_dir

    def plot_design_matrix(self, design_matrix, prefix="", out_dir=None):
        if out_dir is None:
            out_dir = self.out_dir + "GLM/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plot_design_matrix(design_matrix)
        name = "design_matrix.png"
        if prefix:
            name = prefix + "_" + name
        plt.savefig(out_dir + name)
        plt.close()

    def plot_contrast_matrix(self, prefix="", out_dir=None):
        if out_dir is None:
            out_dir = self.out_dir + "GLM/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if isinstance(self.contrasts, dict):
            for contrast_name, contrast_matrix in self.contrasts.items():
                plot_contrast_matrix(contrast_matrix)
                name = f"{contrast_name}_contrast_matrix.png"
                if prefix:
                    name = prefix + "_" + name
                plt.savefig(out_dir + name)
                plt.close()
        elif isinstance(self.contrasts, list):
            if isinstance(self.contrasts[0], str):
                raise TypeError("Can only plot matrix not contrast name")
            else:
                raise NotImplementedError(
                    "Only supports the plotting of contrast matrix"
                )

    def loop_through_contrasts(self, fmri_glm, output_prefix):
        return_dict = dict()
        if isinstance(self.contrasts, dict):
            conditions = self.contrasts.keys()
            con_matrix_all = self.contrasts.values()
        elif isinstance(self.contrasts, list):
            conditions = self.contrasts
            con_matrix_all = [None for i in range(len(conditions))]
        else:
            raise TypeError(
                "Contrasts must be a list of conditions or a dictionary of conditions and contrast matrices"
            )
        for condition, con_matrix in zip(conditions, con_matrix_all):
            effect_fname = f"{output_prefix}_{condition}_effect.nii.gz"
            z_fname = f"{output_prefix}_{condition}_z.nii.gz"
            p_value_fname = f"{output_prefix}_{condition}_p.nii.gz"
            if con_matrix is None:
                contrast = condition
            else:
                contrast = con_matrix
            fmri_glm.compute_contrast(contrast, output_type="effect_size").to_filename(
                os.path.join(self.out_dir, effect_fname)
            )

            fmri_glm.compute_contrast(contrast, output_type="z_score").to_filename(
                os.path.join(self.out_dir, z_fname)
            )

            fmri_glm.compute_contrast(contrast, output_type="p_value").to_filename(
                os.path.join(self.out_dir, p_value_fname)
            )
            return_dict[condition] = {
                "effect": self.out_dir + effect_fname,
                "z": self.out_dir + z_fname,
                "p": self.out_dir + p_value_fname,
            }

        return return_dict

    def plot_single_stat(
        self, stat_map, out_name=None, plot="3D", plot_kwargs=None, save=True
    ):
        if plot_kwargs is None and plot == "3D":
            plot_kwargs = {
                "threshold": 3.0,
                "display_mode": "ortho",
                "black_bg": False,
                "title": out_name,
            }
        elif plot_kwargs is None and plot == "glass":
            plot_kwargs = {
                "threshold": 3.0,
                "colorbar": "False",
                "black_bg": False,
                "plot_abs": False,
                "display_mode": "z",
            }
        elif plot_kwargs is None and plot == "surface":
            plot_kwargs = {
                "threshold": 3.0,
                "colorbar": "False",
                "black_bg": False,
                "plot_abs": False,
                "display_mode": "z",
            }

        if plot == "3D":
            plot_stat_map(
                stat_map,
                **plot_kwargs,
            )
        elif plot == "glass":
            plot_glass_brain(
                stat_map,
                **plot_kwargs,
            )
        elif plot == "surface":
            view_img_on_surf(
                stat_map,
                **plot_kwargs,
            )

        if save:
            plt.savefig(os.path.join(self.out_dir, f"{out_name}_zmap.png"))

            # view.save_as_html(
            #     os.path.join(out_dir, f"group_{condition}_{contrast}_zmap.html")
            # )
        plt.close()

    # TODO: Add thresholding
    def thresh_stat(
        self,
    ):
        pass

    def get_tables(self):
        pass

    def plot_all_stat(
        self, list_stat_maps: list, out_name_prefix, plot_kwargs=None, save=True
    ):
        for idx, stat_map in enumerate(list_stat_maps):
            if idx % 25 == 0:
                if idx != 0 and save:
                    plt.savefig(
                        os.path.join(
                            self.out_dir, f"{out_name_prefix}_zmap_{idx % 25 + 1}.png"
                        )
                    )
                    plt.close()
                fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))
                axes = axes.ravel()
                cidx = 0

            plot_params = {
                "title": os.path.basename(stat_map),
                "axes": axes[cidx],
                "plot_abs": False,
                "display_mode": "z",
            }
            plot_params.update(plot_kwargs)
            self.plot_single_stat(
                stat_map, plot="glass", plot_kwargs=plot_kwargs, save=False
            )
            cidx += 1

    @staticmethod
    def _load_csv(csv_file):
        if isinstance(csv_file, str):
            with open(csv_file, "r") as file:
                dialect = csv.Sniffer().sniff(file.read(1024))
                file.seek(0)
                return pd.read_csv(csv_file, delimiter=dialect.delimiter)
        elif isinstance(csv_file, pd.DataFrame):
            return csv_file


class FirstLevelPipe(TaskFmri):
    def __init__(
        self,
        tr: float or int,
        contrasts: List[str] or dict,
        out_dir: str,
        imgs: List[bids.layout.models.BIDSImageFile] = None,
        confounds: List[str or pd.DataFrame] = None,
        confound_items: List[str] = None,
        events: List[str or pd.DataFrame] = None,
        prep_func: str or Callable or None = None,
        first_level_kwargs: dict = None,
    ):
        """

        Note: The imgs, confounds, events can be empty here, in that case they must be specified in the fit method.
        """
        super().__init__(tr, contrasts, out_dir)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.tr = tr
        self.imgs = imgs
        self.confounds = confounds
        self.events = events
        self.first_level_kwargs = {
            "t_r": self.tr,
            "noise_model": "ar1",
            "hrf_model": "spm",
            "drift_model": "cosine",
            "high_pass": 1.0 / 160,
            "signal_scaling": False,
            "minimize_memory": True,
        }
        self.prep_func = prep_func
        self.confound_items = confound_items
        self.first_level_kwargs.update(first_level_kwargs)
        self.to_second_level = {}
        if isinstance(contrasts, dict):
            for key, value in contrasts.items():
                self.to_second_level[key] = []
        elif isinstance(contrasts, list):
            for contrast in contrasts:
                self.to_second_level[contrast] = []
        else:
            raise ValueError("Contrasts must be either dict or list")

    def process_subject(
        self,
        img: nib.nifti1.Nifti1Image,
        confound: pd.DataFrame,
        event: pd.DataFrame,
        out_prefix: str,
        plot_design: bool = True,
    ):
        event = event[["trial_type", "onset", "duration"]]
        fmri_glm = FirstLevelModel(**self.first_level_kwargs)
        fmri_glm = fmri_glm.fit(img, events=event, confounds=confound)  # fit the model
        design_matrix = fmri_glm.design_matrices_[0]
        design_validity = self.__check_design(design_matrix)
        if design_validity == "valid":
            if plot_design:
                self.plot_design_matrix(design_matrix, out_prefix)
        else:
            print(f"Design matrix for {out_prefix} is {design_validity}")
        del img  # release the memory
        subj_results = self.loop_through_contrasts(fmri_glm, out_prefix)

        return subj_results

    def loop_through_subjects(
        self,
        out_prefixes: List[str] = None,
    ):
        if self.confound_items is None:
            self.confound_items = [
                "white_matter",
                "framewise_displacement",
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
            ]
        i = 0
        for img, confound_name, event_name in tqdm(
            zip(self.imgs, self.confounds, self.events)
        ):
            if isinstance(img, bids.layout.models.BIDSImageFile):
                out_prefix = img.get_entities()["subject"]
                img = img.get_image()
            else:
                out_prefix = f"sub-{i}"
                img = load_imgs(img)
            if out_prefixes is not None:
                out_prefix = out_prefixes[i]
            tqdm.write(f"Processing subject {out_prefix}")
            if isinstance(self.prep_func, Callable):
                img = self.prep_func(img)
            elif self.prep_func == "default":
                img = self.prep_img(img)
            elif isinstance(self.prep_func, str):
                img = self.prep_img(img)
            elif self.prep_func is None:
                pass
            if confound_name is None:
                confound = None
            else:
                confound = self._load_csv(confound_name)[self.confound_items]
            event = self._load_csv(event_name)
            subj_results = self.process_subject(img, confound, event, out_prefix)
            for contrast, imgs in subj_results.items():
                self.to_second_level[contrast].append(imgs["z"])
            i += 1

    def prep_img(self, img, **kwargs):
        default_kwargs = {
            "smoothing_fwhm": 6.0,
            "standardize": True,
            "detrend": True,
            "low_pass": 0.1,
            "high_pass": 0.01,
            "confounds": None,
        }
        default_kwargs.update(kwargs)
        preprocessed_img = clean_img(img, t_r=self.tr, **default_kwargs)
        return preprocessed_img

    @staticmethod
    def __check_design(design):
        # TODO: check if design is valid
        frame_times, matrix, names = check_design_matrix(design)
        return "valid"

    def fit(self, X, y=None):
        # X is expected to be a tuple containing (imgs, confounds, events)
        # You may need to modify this depending on your specific input structure
        imgs, confounds, confound_items, events = X

        self.imgs = imgs
        self.confounds = confounds
        self.events = events
        self.confound_items = confound_items
        self.loop_through_subjects()

        return self

    def transform(self, X, y=None):
        # You can return the `to_second_level` attribute here,
        # as it contains the results of first-level analysis
        return self.to_second_level


class HigherLevelPipe(TaskFmri):
    def __init__(
        self,
        tr: float or int,
        design_matrix: pd.DataFrame,
        contrasts: List[str] or dict,
        out_dir: str,
        stat_maps: dict = None,
        non_parametric: bool = False,
        higher_level_kwargs: dict = None,
    ):
        """
        Higher level analysis pipeline

        Note: stat_maps is a dict containing the first level results, it can be empty when initializing the class
        In that case it must be specified in the fit method

        Parameters
        ----------
        tr
        design_matrix
        contrasts
        out_dir
        stat_maps
        non_parametric
        higher_level_kwargs
        """
        super().__init__(tr, contrasts, out_dir)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.stat_maps = stat_maps
        self.design_matrix = design_matrix
        self.higher_results = {}
        if isinstance(contrasts, dict):
            for key, value in contrasts.items():
                self.higher_results[key] = []
        elif isinstance(contrasts, list):
            for contrast in contrasts:
                self.higher_results[contrast] = []
        else:
            raise ValueError("Contrasts must be either dict or list")
        self.nonparametric = non_parametric
        self.model_kwargs = {}
        self.model_kwargs.update(higher_level_kwargs)

    def process_single_1level_contrast(self, stat_maps, out_prefix):
        if self.nonparametric:
            second_level_model = self.loop_through_contrasts_nonparametric(
                stat_maps, out_prefix
            )
        else:
            second_level_model = SecondLevelModel(**self.model_kwargs)
            second_level_model = second_level_model.fit(
                stat_maps, design_matrix=self.design_matrix
            )
            del stat_maps
            higher_level_results = self.loop_through_contrasts(
                second_level_model, out_prefix
            )
        return higher_level_results

    def loop_all_1level_contrasts(self):
        for contrast_1level, stat_maps in tqdm(self.stat_maps.items()):
            tqdm.write(f"Processing contrast {contrast_1level}")
            out_prefix = contrast_1level
            stat_maps = load_imgs(stat_maps)
            group_results = self.process_single_1level_contrast(stat_maps, out_prefix)
            for contrast, imgs in group_results.items():
                self.higher_results[contrast].append(imgs["z"])

        return self.higher_results

    def loop_through_contrasts_nonparametric(self, stat_maps, output_prefix):
        """
        References: https://nilearn.github.io/dev/modules/generated/nilearn.glm.second_level.non_parametric_inference.html#nilearn.glm.second_level.non_parametric_inference
        """
        return_dict = dict()
        if isinstance(self.contrasts, dict):
            conditions = self.contrasts.keys()
            con_matrix_all = self.contrasts.values()
        elif isinstance(self.contrasts, list):
            conditions = self.contrasts
            con_matrix_all = [None for i in range(len(conditions))]
        else:
            raise TypeError(
                "Contrasts must be a list of conditions or a dictionary of conditions and contrast matrices"
            )
        for condition, con_matrix in zip(conditions, con_matrix_all):
            if con_matrix is None:
                contrast = condition
            else:
                contrast = con_matrix
            if self.model_kwargs["tfce"] or self.model_kwargs["threshold"]:
                out_dic = non_parametric_inference(
                    stat_maps,
                    design_matrix=self.design_matrix,
                    second_level_contrast=contrast,
                    **self.model_kwargs,
                )
                return_dict[condition] = out_dic
                self.plot_cluster_results(out_dic, output_prefix + "_" + condition)
            else:
                neg_log10_vfwe_pvals_img = non_parametric_inference(
                    stat_maps,
                    design_matrix=self.design_matrix,
                    second_level_contrast=contrast,
                    **self.model_kwargs,
                )
                return_dict[condition] = neg_log10_vfwe_pvals_img
                # save img
                neg_log10_vfwe_pvals_img.to_filename(
                    f"{output_prefix}_{condition}_FWER-corrected_p.nii.gz"
                )

        return return_dict

    def plot_cluster_results(self, results: dict, output_prefix):
        threshold = -np.log(0.05)
        vmax = -np.log(1 / self.model_kwargs["n_perm"])

        images = []
        titles = []
        for img_name, img in results.items():
            img.to_filename(f"{output_prefix}_{img_name}.nii.gz")
            if img_name in [
                "logp_max_t",
                "logp_max_size",
                "logp_max_mass",
                "logp_max_tfce",
            ]:
                images.append(results[img_name])
            if img_name == "logp_max_t":
                titles.append("Permutation Test\n(Voxel-Level Error Control)")
            elif img_name == "logp_max_size":
                titles.append("Permutation Test\n(Cluster-Size Error Control)")
            elif img_name == "logp_max_mass":
                titles.append("Permutation Test\n(Cluster-Mass Error Control)")
            elif img_name == "logp_max_tfce":
                titles.append("Permutation Test\n(TFCE Error Control)")

        fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
        axes = axes.ravel()
        for img_counter, ax in enumerate(axes):
            if img_counter >= len(images):
                break
            plot_glass_brain(
                images[img_counter],
                colorbar=True,
                vmax=vmax,
                display_mode="z",
                plot_abs=False,
                threshold=threshold,
                figure=fig,
                axes=ax,
            )
            ax.set_title(titles[img_counter])
        fig.suptitle("Higher level results")
        fig.savefig(f"{output_prefix}_higher_level_results.png")
        plt.close()

    def fit(self, X, y=None):
        # X is expected to be a dictionary containing {contrast: [stat_maps]}
        self.stat_maps = X
        self.loop_all_1level_contrasts()

        return self

    def transform(self, X, y=None):
        # Return the `higher_results` attribute here,
        # as it contains the results of the second-level analysis
        return self.higher_results


def load_imgs(
    imgs: list or str or nib.nifti1.Nifti1Image or bids.layout.models.BIDSImageFile,
):
    if isinstance(imgs, str):
        return nib.load(imgs)
    elif isinstance(imgs, nib.nifti1.Nifti1Image):
        return imgs
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
