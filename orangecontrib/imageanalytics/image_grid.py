# import importlib

import numpy as np
from Orange.data import Table, Instance
from Orange.distance import Cosine
from Orange.preprocess import Normalize
from Orange.projection import MDS, PCA, TSNE
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis

# try:
#     from lap import lapjv
# except ImportError:
from scipy.optimize import linear_sum_assignment


class ImageGrid:
    def __init__(self, data):
        self.size_x, self.size_y = 0, 0
        self.cost, self.grid_indices, self.assignments, self.image_list = 0, [], [], []
        self.data = data

        # reduce dimensions
        data_2dim = Table(self._reduce_dimensions(data))

        # normalize the data
        self.norm_data = np.array(self._normalize_data(data_2dim))

    # def process(self, size_x=0, size_y=0):
    #     """
    #     Process the data based on the provided grid size.
    #     If the desired grid size is not provided, it is calculated from the data.
    #
    #     This method is run every time the grid size changes.
    #
    #     Parameters
    #     ----------
    #     size_x: int
    #         The number of columns.
    #     size_y: int
    #         The number of rows.
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     # check if grid will fit all of the images
    #     if size_x and size_y:
    #         assert size_x * size_y >= len(self.data)
    #         self.size_x, self.size_y = size_x, size_y
    #
    #     # calculate the necessary grid size
    #     else:
    #         self.size_x, self.size_y = self._get_grid_size(self.norm_data, use_default_square=False)
    #
    #     # calculate grid assignments and cost of the assignment
    #     self.cost, self.grid_indices, self.assignments = self._get_assignments(self.norm_data)
    #
    #     # multiply by cell size to get [0, n) indices for grid
    #     self.assignments.T[0] *= self.size_x
    #     self.assignments.T[1] *= self.size_y
    #
    #     self.image_list = self._grid_indices_to_image_list(self.data)

    @staticmethod
    def _reduce_dimensions(data, method="MDS", use_cosine=False):
        """
        Reduce the dimensionality of the data to 2D.

        Parameters
        ----------
        data: Orange.data.Table
            The image embeddings (vectors of length 2048).
        method: string
            The method to use (default MDS).
        use_cosine: bool
            Precompute cosine distances and pass them to MDS.

        Returns
        -------
        array-like
            The data, reduced to 2 dimensions.

        """
        if method == "MDS":
            if use_cosine:
                mds = MDS(n_init=1, dissimilarity="precomputed")
                dist_matrix = Cosine(data)
                return mds(dist_matrix).embedding_
            else:
                mds = MDS(n_init=1, init_type="PCA")
                return mds(data).embedding_

        elif method == "PCA":
            pca = PCA(n_components=2)
            return pca(data)(data)

        elif method == "TSNE":
            tsne = TSNE(init="pca")
            return tsne(data).embedding_

    # @staticmethod
    # def _get_grid_size(data, use_default_square=False):
    #     """
    #     Calculate the size of the grid.
    #
    #     Parameters
    #     ----------
    #     data: array-like
    #         The normalized data.
    #     use_default_square: bool
    #         Define the grid as the minimal possible square.
    #
    #     Returns
    #     -------
    #     int, int
    #         The width and height of the grid.
    #
    #     """
    #
    #     # if the grid would be square, this is the minimum size
    #     sqr_size = int(np.ceil(np.sqrt(len(data))))
    #     size_x = size_y = sqr_size
    #
    #     if not use_default_square:
    #         kurt = kurtosis(data)
    #         kurt_x, kurt_y = np.int32(np.abs(np.ceil(kurt * 2)))
    #         size_x += kurt_x
    #         size_y += kurt_y
    #
    #     return size_x, size_y

    @staticmethod
    def _normalize_data(data):
        """
         Normalize the data (a series of 2D coordinates) to the interval [0, 1].

        Parameters
        ----------
        data: Orange.data.Table
            The data to be normalized.

        Returns
        -------
        Orange.data.Table
            The normalized data constrained to [0, 1].
        """

        normalizer = Normalize(norm_type=Normalize.NormalizeBySpan)
        return normalizer(data)

    # def _get_assignments(self, data):
    #     """
    #     Get the assignments of the 2D points to the regular grid by solving the linear assignment problem
    #     using the Jonker-Volgenant algorithm.
    #
    #     Parameters
    #     ----------
    #     data: np.ndarray
    #         The normalized data.
    #
    #     Returns
    #     -------
    #     int, array-like, array-like
    #         The cost and assigments to grid cells.
    #
    #     """
    #     # create grid of size n with linearly spaced coordinates, reshape to list of coordinates (size_x*size_y x2)
    #     grid = np.dstack(np.meshgrid(np.linspace(0, 1, self.size_x, endpoint=False),
    #                                  np.linspace(0, 1, self.size_y, endpoint=False))).reshape(-1, 2)
    #
    #     # get squared euclidean distances between all pairs of grid points and embeddings
    #     cost_matrix = cdist(grid, data, "sqeuclidean")
    #
    #     """
    #     calculate the linear assignment problem - find the assignments with minimal cost
    #     the algorithm works if we have less embeddings than there are grid cells (generalized LAP)
    #     -1 is returned if there is no matching embedding, i.e. no samples match the grid cell
    #     """
    #
    #     # Try to use lap if it is available - if not, use scipy's linear_sum_assignment.
    #     # if importlib.util.find_spec("lap") is not None:
    #     #     res = lapjv(cost_matrix, extend_cost=True)
    #     #     cost, grid_indices, assignments = res[0], res[1], grid[res[2]]
    #     # else:
    #     row_indices, col_indices = linear_sum_assignment(cost_matrix)
    #     cost = cost_matrix[row_indices, col_indices].sum()
    #
    #     grid_indices = np.full((self.size_x * self.size_y), -1, dtype=np.int32)
    #     grid_indices[row_indices] = col_indices
    #     assignments = grid[row_indices]
    #
    #     return cost, grid_indices, assignments
    #
    # def _grid_indices_to_image_list(self, images):
    #     """
    #     Return the image grid as a Table of images, ordered by rows.
    #     If a grid cell does not contain an image, put None in its place.
    #
    #     Parameters
    #     ----------
    #     images: Orange.data.Table
    #         The images to order.
    #
    #     Returns
    #     -------
    #     Orange.data.Table
    #         A Table of images in the grid, ordered by rows.
    #     """
    #
    #     image_list = Table.from_domain(images.domain)
    #     image_list.extend([Instance(images.domain, image) for image in self.order_to_grid(images)])
    #     return image_list
    #
    # def order_to_grid(self, data):
    #     """
    #     Order the data to grid cells according to the grid indices.
    #     If a grid cell does not contain data, put None in its place.
    #
    #     Parameters
    #     ----------
    #     data: array-like
    #         The data to order.
    #
    #     Returns
    #     -------
    #     array-like
    #         A list of the provided data in the grid, ordered by rows.
    #     """
    #     return [None if i == -1 else data[i] for i in self.grid_indices]
