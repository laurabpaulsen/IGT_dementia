import stan
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

# local  imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.fit_model import fit_group_level_one_group
from utils.helper_functions import parse_n_subj_groups


if __name__ == "__main__":
    path = Path(__file__).parent

    n_subj, n_groups = parse_n_subj_groups()

    inpath = path / "simulated" / "group_lvl" / f"{n_groups}" / f"{n_subj}"
    outpath = path / "fit" / "group_lvl_one_group"  / f"{n_groups}" / f"{n_subj}"

    if not outpath.exists():
        outpath.mkdir(parents = True)

    with open(path.parent / "hierachical_IGT_ORL_one_group.stan") as f:
        model_spec = f.read()

 
    for group in range(1, n_groups + 1):
        # group outpath
        outpath_group = outpath / f"param_rec_{group}.csv"

        # check if there is already a file for this comparison, if so, skip!
        if outpath_group.exists():
            print(f"group {group} already exists, skipping...")
            continue

        print(f"Recovering group {group}")
        # save an empty file so that other processes know that this comparison is being worked on
        outpath_group.touch()

        
        filename = f"ORL_simulated_group_{group}.csv"
        data = pd.read_csv(inpath / filename)
        
        
        fit_group_level_one_group(
            data = data,
            model_spec = model_spec,
            savepath = outpath / f"param_rec_{group}.csv"
        )
        print(f"Finished recovering group {group}")