import os
from tqdm import tqdm
from typing import Callable, List, Dict
from neuroimager.utils.rbload import rbload_csv, rbload_imgs

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import ndarray

from sklearn.base import BaseEstimator, TransformerMixin
import nibabel as nib
import bids.layout.models

from nilearn.glm.first_level import FirstLevelModel, check_design_matrix
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img
from nilearn.image import clean_img
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.plotting import plot_stat_map, view_img_on_surf, plot_glass_brain
from nilearn.reporting import get_clusters_table


# TODO: Check the input arguments for better debugging
class TaskFmri(BaseEstimator, TransformerMixin, object):
    def __init__(
        self,
        tr: int,
        contrasts: list or dict,
        out_dir: str,
        generate_report: bool = False,
    ):
        self.tr = tr
        self.contrasts = contrasts
        self.out_dir = out_dir
        self.generate_report = generate_report

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

    def loop_all_contrasts_parametric(self, fmri_glm, output_prefix):
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

    def generate_nilearn_report(
        self,
        fitted_glm: FirstLevelModel or SecondLevelModel,
        contrasts: Dict[str, ndarray] or str or List[str] or ndarray or List[ndarray],
        title=None,
        bg_img="MNI152TEMPLATE",
        threshold=1.96,
        alpha=0.05,
        cluster_threshold=10,
        height_control="fpr",
        min_distance=8.0,
        plot_type="glass",
        display_mode="lyrz",
        report_dims=(1600, 800),
    ):
        html = fitted_glm.generate_report(
            contrasts,
            title=title,
            bg_img=bg_img,
            threshold=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            min_distance=min_distance,
            plot_type=plot_type,
            display_mode=display_mode,
            report_dims=report_dims,
        )
        with open(self.out_dir + f"{title}report.html", "w") as f:
            f.write(html)

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


class FirstLevelPipe(TaskFmri):
    def __init__(
        self,
        tr: float or int,
        contrasts: List[str] or dict,
        out_dir: str,
        imgs: List[bids.layout.models.BIDSImageFile] = None,
        subj_ids: List[str] = None,
        confounds: List[str or pd.DataFrame] = None,
        confound_items: List[str] = None,
        events: List[str or pd.DataFrame] = None,
        prep_func: str or Callable or None = None,
        generate_report: bool = False,
        first_level_kwargs: dict = None,
    ):
        """

        Note: The imgs, confounds, events can be empty here, in that case they must be specified in the fit method.
        """
        super().__init__(tr, contrasts, out_dir, generate_report)
        self.out_dir = out_dir
        self.tr = tr
        self.imgs = imgs
        self.subj_ids = subj_ids
        self.confounds = confounds
        self.confound_items = confound_items
        self.events = events
        self.prep_func = prep_func
        self.first_level_kwargs = {
            "t_r": self.tr,
            "noise_model": "ar1",
            "hrf_model": "spm",
            "drift_model": "cosine",
            "high_pass": 1.0 / 160,
            "signal_scaling": False,
            "minimize_memory": True,
        }
        self.first_level_kwargs.update(first_level_kwargs)
        self.to_second_level = {}
        self.__prep_input()

    def __prep_input(self):
        os.makedirs(self.out_dir, exist_ok=True)
        if self.imgs is not None:
            if self.subj_ids is not None:
                if len(self.imgs) != len(self.subj_ids):
                    raise ValueError(
                        "The number of images and subject ids must be equal"
                    )
            if confounds is None:
                self.confounds = [None] * len(imgs)
        if self.confounds is not None:
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
        if isinstance(self.contrasts, dict):
            for key, value in self.contrasts.items():
                self.to_second_level[key] = []
        elif isinstance(self.contrasts, list):
            for contrast in self.contrasts:
                self.to_second_level[contrast] = []
        else:
            raise ValueError("Contrasts must be either dict or list")

    def process_single_subject(
        self,
        img: nib.nifti1.Nifti1Image,
        confound: pd.DataFrame,
        event: pd.DataFrame,
        out_prefix: str,
        plot_design: bool = True,
    ):
        event = event[["trial_type", "onset", "duration"]]
        first_level_model = FirstLevelModel(**self.first_level_kwargs)
        first_level_model = first_level_model.fit(
            img, events=event, confounds=confound
        )  # fit the model
        if self.generate_report:
            self.generate_nilearn_report(first_level_model, self.contrasts, out_prefix)
        design_matrix = first_level_model.design_matrices_[0]
        design_validity = self.__check_design(design_matrix)
        if design_validity == "valid":
            if plot_design:
                self.plot_design_matrix(design_matrix, out_prefix)
        else:
            print(f"Design matrix for {out_prefix} is {design_validity}")
        del img  # release the memory
        subj_results = self.loop_all_contrasts_parametric(first_level_model, out_prefix)

        return subj_results

    def loop_all_subjects(
        self,
        out_prefixes: List[str] = None,
    ):
        """
        Conduct first level analysis for all subjects
        Parameters
        ----------
        out_prefixes: List[str] By default, the subject id is used as the prefix for the output files.
            If the input images are not bids, then the prefix is sub-0, sub-1, etc. Or it will be the subject id if specified.
            If out_prefixes is specified, then it will be used as the prefix for the output files.
            out_prefixes > self.subj_ids > img.get_entities()["subject"] > sub-0, sub-1, etc.

        Returns
        -------

        """
        i = 0
        for img, confound_name, event_name in tqdm(
            zip(self.imgs, self.confounds, self.events)
        ):
            if isinstance(img, bids.layout.models.BIDSImageFile):
                out_prefix = img.get_entities()["subject"]
                img = img.get_image()
            else:
                out_prefix = f"sub-{i}"
                img = rbload_imgs(img)[0]
            if self.subj_ids is not None:
                out_prefix = self.subj_ids[i]
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
                confound = rbload_csv(confound_name)[self.confound_items]
            event = rbload_csv(event_name)
            subj_results = self.process_single_subject(img, confound, event, out_prefix)
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
        """
        Parameters
        ----------
        X: X is expected to be a tuple containing (imgs, confounds,confound_items, events)
            confound_items is meaningful only if confounds is not None
        y: For consistency with scikit-learn API

        Returns
        -------
        self
        """
        # You may need to modify this depending on your specific input structure
        imgs, confounds, confound_items, events = X

        self.imgs = imgs
        self.events = events
        self.confound_items = confound_items
        self.confounds = confounds
        self.__prep_input()
        self.loop_all_subjects()

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
        generate_report: bool = True,
        stat_map_masks: list or dict = None,
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
        super().__init__(tr, contrasts, out_dir, generate_report)
        self.out_dir = out_dir
        self.stat_maps = stat_maps
        self.design_matrix = design_matrix
        self.higher_results = {}
        self.nonparametric = non_parametric
        self.stat_map_masks = stat_map_masks
        self.model_kwargs = {}
        self.model_kwargs.update(higher_level_kwargs)
        self.__prep_input()

    def __prep_input(self):
        os.makedirs(self.out_dir, exist_ok=True)
        if isinstance(self.contrasts, dict):
            for key, value in self.contrasts.items():
                self.higher_results[key] = []
        elif isinstance(self.contrasts, list):
            for contrast in self.contrasts:
                self.higher_results[contrast] = []
        else:
            raise ValueError("Contrasts must be either dict or list")

    def process_single_1level_contrast(self, stat_maps, out_prefix, plot=True):
        if self.nonparametric:
            higher_level_result = self.loop_all_2level_contrasts_nonparametric(
                stat_maps, out_prefix
            )
        else:
            second_level_model = SecondLevelModel(**self.model_kwargs)
            second_level_model = second_level_model.fit(
                stat_maps, design_matrix=self.design_matrix
            )
            if self.generate_report:
                self.generate_nilearn_report(
                    second_level_model, self.contrasts, out_prefix
                )
            del stat_maps
            higher_level_result = self.loop_all_2level_contrasts_parametric(
                second_level_model, out_prefix
            )
        if plot:
            self.plot_higher_level_result(higher_level_result, out_prefix)

        return higher_level_result

    def plot_higher_level_result(self, higher_level_result, out_prefix):
        # TODO: Update docstrings.
        # TODO: add **kwargs support for all plotting functions
        for contrast_1level, returned_dict in higher_level_result.items():
            if self.nonparametric:
                if isinstance(returned_dict, dict):
                    self.plot_cluster_nonparametric(
                        returned_dict, out_prefix + "_" + contrast_1level
                    )
                else:
                    img = returned_dict
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plot_glass_brain(
                        img,
                        colorbar=True,
                        display_mode="ortho",
                        plot_abs=False,
                        figure=fig,
                        axes=ax,
                    )
                    fig.suptitle(
                        "Higher level nonparametric results (Displaying FWER-corrected Negative log10 p-values)"
                    )
                    # TODO: save the fig to proper location
                    # fig.savefig()
            else:
                if self.stat_map_masks is None:
                    masks = [None]
                    masks_names = [None]
                elif isinstance(self.stat_map_masks, list):
                    masks = self.stat_map_masks
                    masks_names = [f"mask{i+1}" for i in range(len(masks))]
                elif isinstance(self.stat_map_masks, dict):
                    masks = list(self.stat_map_masks.values())
                    masks_names = list(self.stat_map_masks.keys())
                else:
                    raise TypeError("stat")

                for mask, mask_name in zip(masks, masks_names):
                    if masks_names[0] is None:
                        plot_prefix = out_prefix + "_" + contrast_1level
                    else:
                        plot_prefix = (
                            out_prefix + "_" + contrast_1level + "_" + mask_name
                        )
                    mask = rbload_imgs(mask)[0]
                    self.plot_cluster_parametric(
                        returned_dict["z"],
                        mask_img=mask,
                        alphas=[0.05, 0.01, 0.001],
                        threshold=1.645,
                        cluster_threshold=0,
                        two_sided=True,
                        save=True,
                        prefix=plot_prefix,
                    )

    def loop_all_1level_contrasts(self):
        for contrast_1level, stat_maps in tqdm(self.stat_maps.items()):
            tqdm.write(f"Processing contrast {contrast_1level}")
            self.plot_all_stat(stat_maps, contrast_1level, save=True)
            out_prefix = contrast_1level
            stat_maps = rbload_imgs(stat_maps)
            group_results = self.process_single_1level_contrast(stat_maps, out_prefix)
            for contrast, imgs in group_results.items():
                self.higher_results[contrast].append(imgs)

        return self.higher_results

    def loop_all_2level_contrasts_parametric(self, fmri_glm, output_prefix):
        """
        THIS FUNCTION IS AN ALIAS OF loop_through_contrasts FOR CLARITY
        """
        higher_level_result = self.loop_all_contrasts_parametric(
            fmri_glm, output_prefix
        )

        return higher_level_result

    def loop_all_2level_contrasts_nonparametric(self, stat_maps, output_prefix):
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
                # self.plot_cluster_nonparametric(out_dic, output_prefix + "_" + condition)
                for key, value in out_dic.items():
                    value.to_filename(f"{output_prefix}_{condition}_{key}.nii.gz")
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

    def plot_cluster_parametric(
        self,
        stat_img,
        mask_img=None,
        alphas: float or List[float] = None,
        threshold: float = 1.96,
        cluster_threshold: float = 10,
        two_sided: bool = True,
        save: bool = True,
        prefix: str = None,
    ):
        """
        Technically this function is exactly same as nilearn.glm.threshold_stats_img.
        This function loops all provided method (None|’fpr’|’fdr’|’bonferroni’) and different alpha level (0.05,0.01,0.001), then save a figure for convenience.

        Parameters
        ----------
        stat_img: Niimg-like object or None, optional
        Statistical image (presumably in z scale) whenever height_control is ‘fpr’ or None, stat_img=None is acceptable. If it is ‘fdr’ or ‘bonferroni’, an error is raised if stat_img is None.

        mask_img: Niimg-like object, optional,
        Mask image

        alphas : float or list, optional
        Number controlling the thresholding (either a p-value or q-value). Its actual meaning depends on the height_control parameter. This function translates alpha to a z-scale threshold. Default=0.001.

        threshold: float, optional
        Desired threshold in z-scale. This is used only if height_control is None. Default=3.0.

        cluster_threshold: float, optional
        cluster size threshold. In the returned thresholded map, sets of connected voxels (clusters) with size smaller than this number will be removed. Default=0.

        two_sided: Bool, optional
        Whether the thresholding should yield both positive and negative part of the maps. In that case, alpha is corrected by a factor of 2. Default=True.

        save:bool, optional
        If True, save all thresholded images to nii objects.

        prefix: str, optional
        Prefix for the outputfile. If save is True, prefix must be specified.

        Returns
        threshed_imgs_dict
        -------

        """
        if save:
            if not prefix:
                raise ValueError("If save is True, prefix must be specified.")
        threshed_imgs_dict = {}
        if isinstance(alphas, float):
            alphas = [alphas]
        elif alphas is None:
            alphas = [0.05, 0.01, 0.001]
        try:
            alphas = list(alphas)
        except:
            raise TypeError("alpha must be iterable if you didn't pass a single value")
        for alpha in alphas:
            threshed_imgs_dict[f"{alpha}"] = {}
            to_plot_tmp = {}
            for method in [None, "fpr", "fdr", "bonferroni"]:
                thresholded_map, threshold = threshold_stats_img(
                    stat_img=stat_img,
                    mask_img=mask_img,
                    alpha=alpha,
                    threshold=threshold,
                    height_control=method,
                    cluster_threshold=cluster_threshold,
                    two_sided=two_sided,
                )
                img_name = "None" if method is None else method
                threshed_imgs_dict[f"{alpha}"][img_name] = thresholded_map
                if save:
                    thresholded_map.to_filename(
                        self.out_dir
                        + f"{prefix}_{img_name}_threshed_zstat_alpha{alpha}.nii.gz"
                    )
                if img_name == "None":
                    to_plot_tmp[f"Z value threshold {threshold} "] = thresholded_map
                elif img_name == "fpr":
                    to_plot_tmp["False Positive Rate (FPR) Corrected"] = thresholded_map
                elif img_name == "fdr":
                    to_plot_tmp[
                        "False Discovery Rate (FDR) Corrected"
                    ] = thresholded_map
                elif img_name == "bonferroni":
                    to_plot_tmp["Bonferroni Corrected"] = thresholded_map

            fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
            axes = axes.ravel()
            for img_counter, (title, img) in enumerate(to_plot_tmp.items()):
                if img_counter >= len(to_plot_tmp.keys()):
                    break
                ax = axes[img_counter]
                plot_glass_brain(
                    img,
                    colorbar=True,
                    display_mode="z",
                    plot_abs=False,
                    threshold=0,
                    figure=fig,
                    axes=ax,
                )
                ax.set_title(title)
            fig.suptitle(
                "Higher level parametric results (Displaying Thresholded Z-scaled statistics )"
            )
            fig.savefig(
                self.out_dir
                + f"{prefix}_higher_level_parametric_results_alpha{alpha}.png"
            )
            plt.close()

        return threshed_imgs_dict

    def plot_cluster_nonparametric(self, results: dict, output_prefix):
        threshold = -np.log(0.05)
        vmax = -np.log(1 / self.model_kwargs["n_perm"])

        images = []
        titles = []
        for img_name, img in results.items():
            img.to_filename(self.out_dir + f"{output_prefix}_{img_name}.nii.gz")
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
        fig.suptitle(
            "Higher level nonparametric results (Displaying Negative log10 p-values)"
        )
        fig.savefig(
            self.out_dir + f"{output_prefix}_higher_level_nonparametric_results.png"
        )
        plt.close()

    def plot_all_stat(
        self,
        list_stat_maps: list,
        out_name_prefix: str,
        plot_kwargs: dict = None,
        save: bool = True,
    ):
        fig, axes, cidx = None, None, None
        for idx, stat_map in enumerate(list_stat_maps):
            if idx % 25 == 0:
                if idx != 0 and save:
                    fig.savefig(
                        os.path.join(
                            self.out_dir, f"{out_name_prefix}_zmaps_{idx // 25}.png"
                        )
                    )
                    plt.close(fig)
                fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))
                axes = axes.ravel()
                cidx = 0

            plot_params = {
                "title": os.path.basename(stat_map),
                "axes": axes[cidx],
                "threshold": 1.96,
                "plot_abs": False,
                "display_mode": "z",
            }
            if plot_kwargs is not None:
                plot_params.update(plot_kwargs)
            self.plot_single_stat(
                stat_map, plot="glass", plot_kwargs=plot_params, save=False
            )
            cidx += 1

        if save and cidx > 0:  # Save the last figure if there are unsaved subplots
            fig.savefig(
                os.path.join(
                    self.out_dir, f"{out_name_prefix}_zmaps_{int((idx // 25) + 1)}.png"
                )
            )
            plt.close(fig)

    def fit(self, X, y=None):
        # X is expected to be a dictionary containing {contrast: [stat_maps]}
        self.stat_maps = X
        self.loop_all_1level_contrasts()

        return self

    def transform(self, X, y=None):
        # Return the `higher_results` attribute here,
        # as it contains the results of the second-level analysis
        return self.higher_results
