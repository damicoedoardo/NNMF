import numpy as np
import scipy.sparse as sps


def invert_dictionary(id_to_index):

    index_to_id = {}

    for id in id_to_index.keys():
        index = id_to_index[id]
        index_to_id[index] = id

    return index_to_id


def estimate_sparse_size(num_rows, topK):
    """
    :param num_rows: rows or colum of square matrix
    :param topK: number of elements for each row
    :return: size in Byte
    """

    num_cells = num_rows*topK
    sparse_size = 4*num_cells*2 + 8*num_cells

    return sparse_size


def seconds_to_biggest_unit(time_in_seconds, data_array=None):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value/conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            if data_array is not None:
                data_array /= conversion_factor[unit_index][1]

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True

    if data_array is not None:
        return new_time_value, new_time_unit, data_array

    else:
        return new_time_value, new_time_unit


def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """

    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)


def reshapeSparse(sparseMatrix, newShape):

    if sparseMatrix.shape[0] > newShape[0] or sparseMatrix.shape[1] > newShape[1]:
        ValueError("New shape cannot be smaller than SparseMatrix. SparseMatrix shape is: {}, newShape is {}".format(
            sparseMatrix.shape, newShape))

    sparseMatrix = sparseMatrix.tocoo()
    newMatrix = sps.csr_matrix(
        (sparseMatrix.data, (sparseMatrix.row, sparseMatrix.col)), shape=newShape)

    return newMatrix

def avgDicts(list_of_dicts):
    """given a non empty list of dicts with the same keys, it returns a dict
    which contains the avg of each dict in the list
    
    Arguments:
        list_of_dicts {list} --
    """
    result_dict = {x:0 for x in list_of_dicts[0].keys()}
    for key in result_dict.keys():
        results_actual_key = [d[key] for d in list_of_dicts]
        result_dict[key] = sum(results_actual_key)/len(results_actual_key)
    return result_dict