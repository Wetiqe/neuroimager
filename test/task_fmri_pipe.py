from neuroimager.pipes.task_fmri import FirstLevelPipe, HigherLevelPipe
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from bids import BIDSLayout

TR = 2
data_path = "//path/to/data/"
layout = BIDSLayout(data_path, derivatives=True)
all_files = layout.get()

first_out = data_path + "derivatives/tfce/first_level/"
higher_out = data_path + "derivatives/tfce/higher_level/"

# select the files in derivatives folder
prep_fmri = layout.get(
    scope="derivatives",
    extension=".nii.gz",
    suffix="bold",
    task="language",
    return_type="object",
)
confounds = layout.get(
    scope="derivatives",
    extension=".tsv",
    suffix="timeseries",
    task="language",
    return_type="filename",
)
events = layout.get(
    scope="raw", extension=".tsv", suffix="events", return_type="filename"
)

suj_num = len(prep_fmri)


# %%
# first level options
def proc_img(img):
    del_volume = 1
    img = img.slicer[:, :, :, del_volume:]
    from nilearn.image import smooth_img

    img = smooth_img(img, fwhm=6)  # smooth the image
    return img


confounds = [pd.read_csv(c, sep="\t").iloc[1:, :] for c in confounds]

confounds_items = [
    "white_matter",
    "framewise_displacement",
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
]

first_contrasts = {
    "matched-unmatched": np.array(
        [
            1.0,
            -1,
        ]
        + [0] * (len(confounds_items) + 6)
    ),
}
first_level_kwargs = {
    "slice_time_ref": 0.0,
    "hrf_model": "spm",  #  'glover',
    "drift_model": "cosine",
    "high_pass": 1.0 / 160,
    "drift_order": 1,
    "fir_delays": [0],
    "min_onset": -24,
    "mask_img": None,
    "target_affine": None,
    "target_shape": None,
    "smoothing_fwhm": None,
    # 'memory': Memory(location=None),
    # 'memory_level': 1,
    "standardize": False,
    "signal_scaling": 0,
    "noise_model": "ar1",
    "verbose": 0,
    "n_jobs": 10,
    "minimize_memory": True,
    "subject_label": None,
    "random_state": 42,
}


# %%
# Higher Level Options
# use these to filter prep_fmri list in order to create design matrix
remove = ["102", "103", "105", "203", "210", "216"]
hcs = [i for i in prep_fmri if i.get_entities()["subject"].startswith("1")]
pts = [i for i in prep_fmri if i.get_entities()["subject"].startswith("2")]
all_img = hcs + pts
higher_design = pd.DataFrame(
    [
        [1] * len(hcs) + [-1] * len(pts),
    ],
    index=["HC-PT"],
).T

higher_contrasts = {
    "HC-PT": np.array(
        [
            1.0,
        ]
    ),
}

non_parametric = True
second_level_kwargs = {
    "non_parametric": {
        "mask": None,
        "smoothing_fwhm": None,
        "model_intercept": True,
        "n_perm": 10000,
        "two_sided_test": False,
        "random_state": 42,
        "n_jobs": 1,
        "verbose": 0,
        "threshold": 0.001,
        "tfce": True,
    },
    "parametric": {
        "mask": None,
        "target_affine": None,
        "target_shape": None,
        "smoothing_fwhm": 6,
        # "memory":Memory(location=None), # from joblib import Memory
        "memory_level": 1,
        "verbose": 6,
        "n_jobs": 1,
        "minimize_memory": True,
    },
}
# %%
task_pipe = Pipeline(
    [
        (
            "first_level",
            FirstLevelPipe(
                tr=TR,
                contrasts=first_contrasts,
                out_dir=first_out,
                prep_func=proc_img,
                first_level_kwargs=first_level_kwargs,
            ),
        ),
        (
            "higher_level",
            HigherLevelPipe(
                tr=TR,
                design_matrix=higher_design,
                contrasts=higher_contrasts,
                non_parametric=non_parametric,
                out_dir=higher_out,
                higher_level_kwargs=second_level_kwargs["non_parametric"],
            ),
        ),
    ]
)

results = task_pipe.fit(
    (all_img, confounds, confounds_items, events),
)
