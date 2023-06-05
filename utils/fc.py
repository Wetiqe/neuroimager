import numpy as np
import math


def filt_fcs(
    fcs, coords, network_col="Network", excludes: list = None, includes: list = None
):
    """Filter FC matrix based on network labels
    :param fcs: A numpy matrix shaped (n_subjs, n_nodes, n_nodes) representing the functional connectivity data.
    :param coords: A DataFrame containing the coordinates and network labels for each node.
    :param network_col: A string representing the column name in the coords DataFrame that contains the network labels. Default is 'Network'.
    :param excludes: A list of network labels to exclude from the filtered FC matrix. Default is None.
    :param includes: A list of network labels to include in the filtered FC matrix. Default is None.
    :return: A filtered numpy matrix containing only the desired network labels.

    This function filters the input functional connectivity matrix based on the specified network labels.
    It can be used to include or exclude specific networks from the analysis.
    """
    if includes is not None and excludes is not None:
        raise ValueError("Only one of 'includes' and 'excludes' can be specified.")
    elif excludes is None:
        excludes = []
    elif includes is None:
        includes = []

    def get_coords_index(coords, col, values):
        return coords[coords[col].isin(values)].index

    if includes:
        ind = get_coords_index(coords, network_col, includes)
        indexer = np.ix_(np.arange(fcs.shape[0]), ind, ind)
        fcs = fcs[indexer]
    elif excludes:
        ind = get_coords_index(coords, network_col, excludes)
        fcs = np.delete(fcs, ind, axis=1)
        fcs = np.delete(fcs, ind, axis=2)

    return fcs


def flatten_lower_triangular(matrix):
    """
    This function takes a 2D or 3D numpy array representing one or multiple square matrices as input and returns a 2D numpy array
    where each row corresponds to the flattened lower triangular part of a matrix in the input. The diagonal and upper triangular
    elements of each input matrix are excluded from the output.

    Parameters:
    matrix (numpy.ndarray): A 2D or 3D numpy array representing one or multiple square matrices.

    Returns:
    numpy.ndarray: A 2D numpy array where each row corresponds to the flattened lower triangular part of a matrix in the input,
    with diagonal and upper triangular elements excluded.
    """
    if len(matrix.shape) == 2:
        matrix = np.array([matrix])
    elif len(matrix.shape) != 3:
        raise ValueError("Input matrix must be 2D or 3D")

    tril_matrices = np.array([np.tril(fc, k=-1) for fc in matrix]).reshape(
        matrix.shape[0], -1
    )
    flat_matrices = tril_matrices[:, ~np.all(tril_matrices == 0, axis=0)]

    return flat_matrices


def unflatten_lower_triangular(flat_matrices, n):
    """
    Reconstructs the lower triangular matrices from their flattened representation.

    Args:
        flat_matrices: A 2D numpy array representing one or multiple flattened lower triangular matrices.
        n: The original size of the matrices.

    Returns:
        A 3D numpy array representing one or multiple lower triangular matrices in their original shape.

    Raises:
        ValueError: If the input matrix is not 2D or has an invalid shape,
                    or if the quadratic equation for `n` has no real solution.
    """
    flat_matrices = np.atleast_2d(flat_matrices)
    if len(flat_matrices.shape) != 2:
        raise ValueError("Input matrix must be 2D")
    m, n = flat_matrices.shape

    def calculate_orig_n(n):
        discriminant = 1 + 8 * n
        if discriminant < 0:
            raise ValueError("No real solution for n")
        else:
            n1 = (1 + math.sqrt(discriminant)) / 2
            n2 = (1 - math.sqrt(discriminant)) / 2
            return n1, n2

    n = int(calculate_orig_n(n)[0])
    tril_indices = np.tril_indices(n, k=-1)
    original_shape_matrices = np.zeros((m, n, n))

    for i in range(m):
        original_shape_matrices[i][tril_indices] = flat_matrices[i, :]

    if m == 1:
        return original_shape_matrices[0]
    else:
        return original_shape_matrices
