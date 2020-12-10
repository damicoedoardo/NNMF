import numpy as np
import RecSysFramework.Utils.compute_popularity as cp


def round_all_digits(number, n_digits):
    return '{:.{prec}f}'.format(number, prec=n_digits)

def get_results_latex_from_dict(d, n_digits=6):
    s = ''
    for cutoff in d.keys():
        d_cutoff = d[cutoff]
        for metric_value in d_cutoff.values():
            s += '& {} '.format(round(metric_value, n_digits))
    return s

def get_items_long_tail_short_head(urm_train, cut_perc=0.66):
    """return the items in the long tail and short head, given a URM train. items belonging to the long tail
    computed in this way: - items sorted by pop
                          - do cum sum
                          - take item i st cum sum is cut_perc*tot_interactions
                          - items < i are long tail, short head otherwise

    Arguments:
        urm_train {csr_matrix} -- 
    
    Returns:
        [(np array,np array)] -- index of items belonging respectively to long tail and short head
    """
    item_pop_tuple_list = cp.compute_popularity_item(urm_train)
    items_idxs, interactions = zip(*item_pop_tuple_list)

    interactions_cumsum = np.cumsum(interactions)
    interactions_cumsum_norm = interactions_cumsum/max(interactions_cumsum)

    cut_idxs = []
    cut_idx = (np.abs(interactions_cumsum_norm - cut_perc)).argmin()
    cut_idxs.append(cut_idx)

    return np.split(items_idxs, cut_idxs)


if __name__ == "__main__":
    from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
    train, _,_,_ = retrieve_train_validation_test_holdhout_dataset()
    get_items_long_tail_short_head(train.get_URM())