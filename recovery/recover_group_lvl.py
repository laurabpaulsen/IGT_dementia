import stan
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

# local  imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.fit_model import fit_group_level


if __name__ == "__main__":
    path = Path(__file__).parent

    inpath = path / "simulated" / "group_lvl"
    outpath = path / "fit" / "group_lvl" 

    if not outpath.exists():
        outpath.mkdir(parents = True)

    with open(path.parent / "hierachical_IGT_ORL.stan") as f:
        model_spec = f.read()

    # make a list of tuples with all combinations of groups (don't want to compare group 1 with group 1 or group 1 with group 2 and group 2 with group 1)
    n_groups = 20
    compare_groups = list(combinations(range(1, n_groups + 1), 2))
 
    for i, (group1, group2) in enumerate(compare_groups):
        filename1 = f"ORL_simulated_group_{group1}_20_sub.csv"
        data1 = pd.read_csv(inpath / filename1)
        
        filename2 = f"ORL_simulated_group_{group2}_20_sub.csv"
        data2 = pd.read_csv(inpath / filename2)
        
        data1["group"] = -0.5
        data2["group"] = 0.5

        data2["sub"] += data1["sub"].max()

        data = pd.concat([data1, data2])
        
        fit_group_level(
            data = data,
            model_spec = model_spec,
            savepath = outpath / f"param_rec_{group1}_{group2}.csv"
        )
        print(f"Finished comparison {i+1} of {len(compare_groups)}")
        print(f"Group {group1} vs group {group2}")