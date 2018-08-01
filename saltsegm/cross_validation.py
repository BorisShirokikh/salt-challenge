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
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        ss = ShuffleSplit(n_splits=1, test_size=val_size, random_state=42)

        tt = []
        for split_tt in kf.split(X=ids):
            tgt.append([ids[split_tt[0]], ids[split_tt[1]]])

        tvt = []
        for split in tt:
            split_t, split_v = next(ss.split(X=split[0]))

            tvt.append({'train_ids': split[0][split_t],
                        'val_ids': split[0][split_v],
                        'test_ids': split[1]})
    else:
        assert len(ids) == len(groups), f'ids and groups should have the same sizes.\
            {len(ids)} and {len(groups)} given.'
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        ss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)

        tgt = []
        for split_tt in kf.split(X=ids, y=groups):
            tgt.append( [ids[split_tt[0]], groups[split_tt[0]], ids[split_tt[1]]] )

        tvt = []
        for split in tgt:
            split_t, split_v = next( ss.split(X=split[0], y=split[1]) )

            tvt.append({'train_ids': split[0][split_t],
                        'val_ids': split[0][split_v],
                        'test_ids': split[2]})

    return tvt
