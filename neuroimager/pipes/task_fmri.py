import os
import gc
import csv

import bids.layout.models
from tqdm import tqdm
from typing import Callable, List, Union, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
from bids import BIDSLayout
from nilearn.image import clean_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.plotting import plot_stat_map, view_img_on_surf, plot_glass_brain


class TaskFmri(object):
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
        plot_contrast_matrix(self.contrasts)
        name = "contrast_matrix.png"
        if prefix:
            name = prefix + "_" + name
        plt.savefig(out_dir + name)
        plt.close()

    def loop_through_contrasts(self, fmri_glm, output_prefix):
        return_dict = dict()
        for condition, con_matrix in self.contrasts.items():
            effect_fname = f"{output_prefix}_{condition}_effect.nii.gz"
            z_fname = f"{output_prefix}_{condition}_z.nii.gz"
            p_value_fname = f"{output_prefix}_{condition}_p.nii.gz"
            fmri_glm.compute_contrast(
                con_matrix, output_type="effect_size"
            ).to_filename(os.path.join(self.out_dir, effect_fname))

            fmri_glm.compute_contrast(con_matrix, output_type="z_score").to_filename(
                os.path.join(self.out_dir, z_fname)
            )

            fmri_glm.compute_contrast(con_matrix, output_type="p_value").to_filename(
                os.path.join(self.out_dir, p_value_fname)
            )
            return_dict[condition] = {
                "effect": effect_fname,
                "z": z_fname,
                "p": p_value_fname,
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
                            self.out_dir, f"{out_name_prefix}_zmap_{idx% 25 + 1}.png"
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
    def __load_csv(csv_file):
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
        tr,
        imgs: List[bids.layout.models.BIDSImageFile],
        confounds: List[str or pd.DataFrame],
        events: List[str or pd.DataFrame],
        contrasts: List[str] or dict,
        out_dir: str,
        first_level_kwargs: dict = None,
    ):
        super().__init__(tr, contrasts, out_dir)
        self.out_dir = out_dir + "/first_level"
        os.makedirs(self.out_dir, exist_ok=True)
        self.imgs = imgs
        self.confounds = confounds
        self.events = events
        self.first_level_kwargs = {
            "noise_model": "ar1",
            "hrf_model": "spm",
            "drift_model": "cosine",
            "high_pass": 1.0 / 160,
            "signal_scaling": False,
            "minimize_memory": True,
        }
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
        confound,
        event,
        out_prefix,
        plot_design=True,
    ):
        fmri_glm = FirstLevelModel(**self.first_level_kwargs)
        fmri_glm = fmri_glm.fit(img, events=event, confounds=confound)  # fit the model
        design_matrix = fmri_glm.design_matrices_[0]
        if plot_design:
            self.plot_design_matrix(design_matrix, out_prefix)
        del img  # release the memory
        subj_results = self.loop_through_contrasts(fmri_glm, out_prefix)

        return subj_results

    def loop_through_subjects(
        self,
        preproc_func: Optional[Callable] or str = None,
        confound_items: List[str] = None,
    ):
        if confound_items is None:
            confound_items = [
                "white_matter",
                "framewise_displacement",
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
            ]
        for bids_img, confound_name, event_name in tqdm(
            zip(self.imgs, self.confounds, self.events)
        ):
            out_prefix = bids_img.get_entities()["subject"]
            tqdm.write(f"Processing subject {out_prefix}")
            img = bids_img.get_image()
            if isinstance(preproc_func, Callable):
                img = preproc_func(img)
            elif preproc_func == "default":
                img = self.prep_img(img)
            elif isinstance(preproc_func, str):
                img = self.prep_img(img)
            confound = self.__load_csv(confound_name)[confound_items]
            event = self.__load_csv(event_name)
            subj_results = self.process_subject(img, confound, event, out_prefix)
            for contrast, imgs in subj_results.items():
                self.to_second_level[contrast].append(imgs["z"])

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


class HigherLevelPipe(TaskFmri):
    def __init__(
        self,
        tr,
        stat_maps: dict,
        contrasts: List[str] or dict,
        out_dir: str,
        second_level_kwargs: dict = None,
    ):
        super().__init__(tr, contrasts, out_dir)
        self.out_dir = out_dir + "/second_level"
        os.makedirs(self.out_dir, exist_ok=True)
        self.imgs = stat_maps
        self.higher_results = {}
        if isinstance(contrasts, dict):
            for key, value in contrasts.items():
                self.higher_results[key] = []
        elif isinstance(contrasts, list):
            for contrast in contrasts:
                self.higher_results[contrast] = []
        else:
            raise ValueError("Contrasts must be either dict or list")
        self.second_level_kwargs = {}
        self.second_level_kwargs.update(second_level_kwargs)

    def process_single_1level_contrast(self, stat_map, out_prefix):
        second_level_model = SecondLevelModel(**self.second_level_kwargs)
        second_level_model = second_level_model.fit(stat_map)
        del stat_map
        higher_level_results = self.loop_through_contrasts(
            second_level_model, out_prefix
        )
        return higher_level_results

    def loop_all_1level_contrasts(self):
        for contrast_1level, stat_maps in tqdm(self.imgs.items()):
            tqdm.write(f"Processing contrast {contrast_1level}")
            out_prefix = contrast_1level
            group_results = self.process_single_1level_contrast(stat_maps, out_prefix)
            for contrast, imgs in group_results.items():
                self.higher_results[contrast].append(imgs["z"])

        return self.higher_results
