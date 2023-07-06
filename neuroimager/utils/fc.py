import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin


# Part one: sklearn estimator types
# These are the wrappers for the functions in part two
class FCFilter(BaseEstimator, TransformerMixin):
    def __init__(self, network_labels, excludes=None, includes=None):
        self.network_labels = network_labels
        self.excludes = excludes
        self.includes = includes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return filter_fcs(X, self.network_labels, self.excludes, self.includes)


class BipartiteMatrixExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, network_labels, bp1_labels, bp2_labels):
        self.network_labels = network_labels
        self.bp1_labels = bp1_labels
        self.bp2_labels = bp2_labels

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return extract_bipartite_matrix(
            X, self.network_labels, self.bp1_labels, self.bp2_labels
        )


class AverageNodesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, labels):
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return average_nodes(X, self.labels)


class FlattenLowerTriangular(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return flatten_lower_triangular(X)


# Part two: functions
def filter_fcs(
    fcs: np.array, network_labels: list, excludes: list = None, includes: list = None
):
    """Filter FC matrix based on network labels
    :param fcs: A numpy matrix shaped (n_subjects, n_nodes, n_nodes) representing the functional connectivity data.
    :param network_labels: A list containing the network labels for each node.
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

    def get_index(labels: list, desired: list):
        return [i for i, label in enumerate(labels) if label in desired]

    if includes:
        ind = get_index(network_labels, includes)
        indexer = np.ix_(np.arange(fcs.shape[0]), ind, ind)
        fcs = fcs[indexer]
    elif excludes:
        ind = get_index(network_labels, excludes)
        fcs = np.delete(fcs, ind, axis=1)
        fcs = np.delete(fcs, ind, axis=2)

    return fcs


def extract_bipartite_matrix(
    matrix: np.array, network_labels: list, bp1_labels: list, bp2_labels: list
):
    """Extract the bipartite part of a functional connectivity matrix.

    :param matrix: A numpy matrix shaped (n_subjs, n_nodes, n_nodes) representing the functional connectivity data.
    :param network_labels: A list containing the network labels for each node.
            should be the same order and length as the number of nodes in the matrix.
    :param bp1_labels: A list of network names for Network 1 nodes.
    :param bp2_labels: A list of network names for other networks' nodes.
    :return: A numpy matrix containing the desired bipartite part of the functional connectivity matrix.
    """
    # the network_labels should have same length as nodes in the matrix
    if len(network_labels) != matrix.shape[1]:
        raise ValueError(
            "The length of network_labels should be equal to the number of nodes in the matrix"
        )

    network1_indices = np.isin(network_labels, bp1_labels)
    other_networks_indices = np.isin(network_labels, bp2_labels)

    bipartite_matrix = matrix[
        np.ix_(range(matrix.shape[0]), network1_indices, other_networks_indices)
    ]

    return bipartite_matrix


def average_nodes(fc_matrix, labels):
    """
    Computes an averaged functional connectivity matrix for each unique label
    in the input labels array. The function accepts both 2D and 3D fc_matrix arrays.

    Parameters:
    -----------
    fc_matrix : numpy array, shape (subject, node, node) or (node, node)
        Functional connectivity matrix or matrices. If 2D, it is assumed to be a single
        matrix. If 3D, each matrix along the first axis corresponds to a subject.

    labels : numpy array, shape (subject,)
        Labels corresponding to each subject in the fc_matrix. Each unique label
        represents a group for which an averaged functional connectivity matrix
        will be computed.

    Returns:
    --------
    avg_fc_matrices : numpy array, shape (unique_labels, node, node)
        Averaged functional connectivity matrices for each unique label in the
        input labels array. If the input fc_matrix is 2D, the output will also
        be 2D with shape (node, node).
    """
    if len(fc_matrix.shape) == 2:
        fc_matrix = np.expand_dims(fc_matrix, axis=0)
    unique_labels = np.unique(np.array(labels))
    n_subjs, n_nodes, _ = fc_matrix.shape
    averaged_fc_matrix = np.zeros((n_subjs, len(unique_labels), len(unique_labels)))

    for i, label in enumerate(unique_labels):
        label_indices = np.where(labels == label)[0]
        for j, other_label in enumerate(unique_labels):
            other_label_indices = np.where(labels == other_label)[0]
            averaged_fc_matrix[:, i, j] = fc_matrix[:, label_indices, :][
                :, :, other_label_indices
            ].mean(axis=(1, 2))

    return averaged_fc_matrix.squeeze()


def flatten_lower_triangular(matrix):
    """
    This function takes a 2D or 3D numpy array returns a 2D numpy array.
    where each row corresponds to the flattened lower triangular part (diagonal removed) of a matrix in the input.

    Parameters:
    matrix (numpy.ndarray): A 2D or 3D numpy array representing one or multiple square matrices.

    Returns:
    numpy.ndarray:  A 2D numpy array representing one or multiple flattened lower triangular matrices (diagonal removed)
    """
    if len(matrix.shape) == 2:
        matrix = np.array([matrix])
    elif len(matrix.shape) != 3:
        raise ValueError("Input matrix must be 2D or 3D")

    tri_matrices = np.array([np.tril(fc, k=-1) for fc in matrix]).reshape(
        matrix.shape[0], -1
    )
    flat_matrices = tri_matrices[:, ~np.all(tri_matrices == 0, axis=0)]

    return flat_matrices


def unflatten_lower_triangular(flat_matrices):
    """
    Reconstructs the lower triangular matrices from their flattened representation.

    Args:
        flat_matrices: A 1D or 2D numpy array representing flattened lower triangular matrices (without diagonal).
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

    def calculate_orig_n(n_nodes):
        discriminant = 1 + 8 * n_nodes
        if discriminant < 0:
            raise ValueError("No real solution for n")
        else:
            n1 = (1 + math.sqrt(discriminant)) / 2
            n2 = (1 - math.sqrt(discriminant)) / 2
            return n1, n2

    n = int(calculate_orig_n(n)[0])
    tri_indices = np.tril_indices(n, k=-1)
    original_shape_matrices = np.zeros((m, n, n))

    for i in range(m):
        original_shape_matrices[i][tri_indices] = flat_matrices[i, :]

    if m == 1:
        return original_shape_matrices[0]
    else:
        return original_shape_matrices
