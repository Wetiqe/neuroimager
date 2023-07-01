# neuroimager

A collection of utilities used for MRI data analysis

# Automatic Analysis of HMM model estimated by HMM-MAR

To be done

# Atlas Operation
Suggest you have two probability atlas:
```python
from nilearn import plotting
import nibabel as nib
import os

# Plot the original prob masks
atlas_path = "./assets/masks/"
files = [
    "HarvardOxford-sub-prob-1mm.nii.gz",
    "JHU-ICBM-tracts-prob-1mm.nii.gz",
]
atlas_paths = [os.path.join(atlas_path, file) for file in files]
# plot listed atlases with nilearn.plotting.plot_prob_atlas
for file in atlas_paths:
    img = nib.load(file)
    plotting.plot_prob_atlas(img, title=file, draw_cross=False, threshold="auto")
    plotting.show()
```
![harvard](./assets/images/harvardoxford.png) 
![jhu](./assets/images/JHU-tracts.png)

Then you can use the atlas_operation to do some operations on the atlases:
```python
## Remove selected regions
```python
from neuroimager.utils import filter_rois

harvard_sub = nib.load("./assets/masks/HarvardOxford-sub-prob-1mm.nii.gz")
rois_to_remove = [0, 1, 11, 12]  # remove the cortical regions in this atlas
output_path = "./assets/output/filtered_harvard_sub.nii.gz"
filtered_atlas = filter_rois(harvard_sub, rois_to_remove, output_path)
plotting.plot_prob_atlas(filtered_atlas, draw_cross=False, threshold="auto")
plotting.show()
```
![filtered_harvard](./assets/images/subregions.png)
## Merge two atlases
```python
from neuroimager.utils import combine_probabilistic_atlases

atlas_paths = [
    "./assets/output/filtered_harvard_sub.nii.gz",
    "./assets/masks/JHU-ICBM-tracts-prob-1mm.nii.gz",
]
combined_atlas = combine_probabilistic_atlases(
    atlas_paths, "./assets/output/combined_prob_atlas.nii.gz"
)
plotting.plot_roi(combined_atlas, title="3D Atlas")
plotting.show()
```
![combined_atlas](./assets/images/combined.png)