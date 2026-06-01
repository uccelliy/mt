import numpy as np
import torch


def pd_to_pth(df, values, keys=['participant', 'task', 'trial']):
    column_names_list = [keys + [value] for value in values]
    wide_arrs = {}
    for column_names in column_names_list:
        arr = df[column_names].values
        dims = [np.unique(arr[:, i], return_inverse=True) for i in range(len(column_names)-1)]
        wide_arr = np.full([len(dims[i][0]) for i in range(len(column_names)-1)], np.nan)
        idx = tuple(dims[i][1] for i in range(len(column_names) - 1))
        wide_arr[idx] = arr[:, -1]
        wide_arrs[column_names[-1]] = torch.from_numpy(wide_arr).reshape(-1, wide_arr.shape[-1])
    return wide_arrs