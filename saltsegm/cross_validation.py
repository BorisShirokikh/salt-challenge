from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def get_cv_111(ids, n_splits=5, val_size=150, groups=None, random_state=42):
    """Splits `ids` on `n_splits` splits with train-val-test strategy.

    Parameters
    ----------
    ids: np.ndarray
        Array of ids to split.

    n_splits: int, optional
        Number of splits of ids to make. Should be `> 1`.

    val_size: int, float, optional
        If `int`, describes number of elements for validation. If `float`,
        describes part of elements for validation.

    groups: np.ndarray, optional
        If `None`, uses simple shuffle split. If array of the same sizes with `ids`,
        uses stratified split strategy.

    random_state: int, optional

    Returns
    -------
    tvt: list
        List of dict(s) which describes cross-val split.
    """
    if groups is None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        ss = ShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)

        # creating main splits into `train` and `test`
        train_test = []
        for split_tt in kf.split(X=ids):
            TRAIN = 0
            TEST = 1

            train_test.append([ids[split_tt[TRAIN]], ids[split_tt[TEST]]])
        # end for

        # each `train` should be splited into `train*` and `val`
        tvt = []
        for split_tt in train_test:
            TRAIN = 0
            TEST = 1

            split_t, split_v = next(ss.split(X=split_tt[TRAIN]))

            tvt.append({'train_ids': split_tt[TRAIN][split_t],
                        'val_ids': split_tt[TRAIN][split_v],
                        'test_ids': split_tt[TEST]})
        # end for
    else:
        assert len(ids) == len(groups), f'ids and groups should have the same sizes.\
            {len(ids)} and {len(groups)} given.'

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        ss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)

        # creating main splits into `train` and `test,
        # also saving `group` for `train` for next `train*` `val` split
        train_group_test = []
        for split_tt in kf.split(X=ids, y=groups):
            TRAIN = 0
            TEST = 1

            train_group_test.append([ids[split_tt[TRAIN]], groups[split_tt[TRAIN]], ids[split_tt[TEST]]])
        # end for

        # each `train` should be splited into `train*` and `val`
        tvt = []
        for split_tgt in train_group_test:
            TRAIN = 0
            GROUP = 1
            TEST = 2

            split_t, split_v = next(ss.split(X=split_tgt[TRAIN], y=split_tgt[GROUP]))

            tvt.append({'train_ids': split_tgt[TRAIN][split_t],
                        'val_ids': split_tgt[TRAIN][split_v],
                        'test_ids': split_tgt[TEST]})
        # end for

    return tvt


def get_cv_11(ids, n_splits=1, val_size=100, groups=None, random_state=42):
    """Splits `ids` on `n_splits` splits with train-val strategy.

    Parameters
    ----------
    ids: np.ndarray
        Array of ids to split.

    n_splits: int, optional
        Number of splits of ids to make.

    val_size: int, float, optional
        If `int`, describes number of elements for validation. If `float`,
        describes part of elements for validation.

    groups: np.ndarray, optional
        If `None`, uses simple shuffle split. If array of the same sizes with `ids`,
        uses stratified split strategy.

    random_state: int, optional

    Returns
    -------
    tv: list
        List of dict(s) which describes cross-val split.
    """
    if groups is None:
        ss = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)

        # creating main splits into `train` and `test`
        tv = []
        for split_tt in ss.split(X=ids):
            TRAIN = 0
            VAL = 1

            tv.append({'train_ids': ids[split_tt[TRAIN]],
                       'val_ids': ids[split_tt[VAL]]})
        # end for
    else:
        assert len(ids) == len(groups), f'ids and groups should have the same sizes.\
            {len(ids)} and {len(groups)} given.'

        ss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)

        # creating main splits into `train` and `test,
        tv = []
        for split_tt in ss.split(X=ids, y=groups):
            TRAIN = 0
            VAL = 1

            tv.append({'train_ids': ids[split_tt[TRAIN]],
                       'val_ids': ids[split_tt[VAL]]})
        # end for

    return tv
