from RecSysFramework.Utils.Common import check_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_popularity_item(urm, ordered=True, plot=False):
    """

    :param urm: urm in which compute the popularity of the items
    :return: ordered list of tuples [(item, interaction_num),....], in ascending order
    """

    urm = check_matrix(urm, 'csr')
    print('Computing item popularity...')

    item_interaction_count_dict = {}
    #iniatialize the entry of each item to 0
    for i in range(urm.shape[1]):
        item_interaction_count_dict[i]=0

    for item in urm.indices:
        item_interaction_count_dict[item] += 1

    l = list(item_interaction_count_dict.items())
    if ordered:
        l = sorted(l, key=lambda x: x[1])

    if plot:
            pop = [(j, l[j][1]) for j in range(len(l))]
            plt.scatter(*zip(*pop))
            plt.show()
    
    return l

def compute_popularity_user(urm, ordered=True, plot=False):
    """

    :param urm: urm in which compute the popularity of the users
    :return: ordered list of tuples [(user, interaction_num),....], in ascending order
    """
    l = []
    for r in range(urm.shape[0]):
        l.append((r, urm.indptr[r+1] - urm.indptr[r]))

    if ordered:
        l = sorted(l, key=lambda x: x[1])

    if plot:
        pop = [(j, l[j][1]) for j in range(len(l))]
        plt.scatter(*zip(*pop))
        plt.show()

    return l
