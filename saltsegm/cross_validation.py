from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def get_cv_111(ids, n_splits=5, val_size=150, groups=None, random_state=42):
    
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
