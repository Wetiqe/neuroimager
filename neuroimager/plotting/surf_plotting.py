import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting


def plot_surf_stat_4views(
    stat_map,
    vmax: int or float = "auto",
    figsize=(12, 12),
    title: str = None,
):
    fsaverage = datasets.fetch_surf_fsaverage()
    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw={"projection": "3d"})
    mesh_right = surface.load_surf_mesh(fsaverage.pial_right)
    mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
    surf_map_right = surface.vol_to_surf(stat_map, mesh_right)
    surf_map_left = surface.vol_to_surf(stat_map, mesh_left)

    if vmax == "auto":
        vmax = max(surf_map_right.max(), surf_map_left.max())
    elif isinstance(vmax, (float, int)):
        pass
    else:
        raise ValueError("vmax must be a number or str")

    # Plot the right hemisphere (lateral and medial views)
    plotting.plot_surf_stat_map(
        mesh_right,
        surf_map_right,
        hemi="right",
        view="lateral",
        axes=axes[0, 1],
        colorbar=False,
        vmax=vmax,
    )
    plotting.plot_surf_stat_map(
        mesh_right,
        surf_map_right,
        hemi="right",
        view="medial",
        axes=axes[1, 1],
        colorbar=False,
        vmax=vmax,
    )

    # Plot the left hemisphere (lateral and medial views)
    plotting.plot_surf_stat_map(
        mesh_left,
        surf_map_left,
        hemi="left",
        view="lateral",
        axes=axes[0, 0],
        colorbar=False,
        vmax=vmax,
    )
    plotting.plot_surf_stat_map(
        mesh_left,
        surf_map_left,
        hemi="left",
        view="medial",
        axes=axes[1, 0],
        colorbar=False,
        vmax=vmax,
    )

    cax = fig.add_axes(
        [0.92, 0.3, 0.02, 0.5]
    )  # Adjust the position and size of the colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plotting.cm.cold_hot, norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=1, wspace=0.0, hspace=-0.4)
    fig.suptitle(title, fontsize=24, y=0.9)

    return fig
