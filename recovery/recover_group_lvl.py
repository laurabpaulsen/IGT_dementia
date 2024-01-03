import stan
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

# local  imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.fit_model import fit_group_level
from utils.helper_functions import parse_n_subj_groups


if __name__ == "__main__":
    path = Path(__file__).parent

    n_subj, n_groups = parse_n_subj_groups()

    inpath = path / "simulated" / "group_lvl" / f"{n_groups}" / f"{n_subj}"
    outpath = path / "fit" / "group_lvl"  / f"{n_groups}" / f"{n_subj}"

    if not outpath.exists():
        outpath.mkdir(parents = True)

    with open(path.parent / "models" / "hierachical_IGT_ORL_compare.stan") as f:
        model_spec = f.read()

    # make a list of tuples with all combinations of groups (don't want to compare group 1 with group 1 or group 1 with group 2 and group 2 with group 1)
    compare_groups = list(combinations(range(1, n_groups + 1), 2))

    # shuffle the list (so that we don't always start with the same comparison) - makes it more fun to plot results as we go along
    np.random.shuffle(compare_groups)
 
    for i, (group1, group2) in enumerate(compare_groups):
        # group outpath
        outpath_group = outpath / f"param_rec_{group1}_{group2}.csv"
    

        # check if there is already a file for this comparison, if so, skip!
        if outpath_group.exists():
            print(f"Comparison {i+1} of {len(compare_groups)} already exists, skipping...")
            continue

        print(f"Starting comparison {group1} vs {group2} ({i+1})")
        # save an empty file so that other processes know that this comparison is being worked on
        outpath_group.touch()

        
        filename1 = f"ORL_simulated_group_{group1}.csv"
        data1 = pd.read_csv(inpath / filename1)
        
        filename2 = f"ORL_simulated_group_{group2}.csv"
        data2 = pd.read_csv(inpath / filename2)


        data1["group"] = 1
        data2["group"] = 2

        data2["sub"] += data1["sub"].max() # make sure that the subject numbers are unique across groups

        data = pd.concat([data1, data2])
        
        fit_group_level(
            data = data,
            model_spec = model_spec,
            savepath = outpath / f"param_rec_{group1}_{group2}.csv"
        )
        print(f"Finished comparison {i+1} of {len(compare_groups)}")
        print(f"Group {group1} vs group {group2}")