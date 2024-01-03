import stan
from pathlib import Path
import numpy as np
import pandas as pd
import argparse


# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.fit_model import fit_group_level_one_group

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--group", required = True, help = "Group to fit the model to.")

    # check if the group is valid
    args = vars(ap.parse_args())
    if args["group"] not in ["HC", "AD"]:
        raise ValueError("Group must be HC or AD.")

    return args["group"]

if __name__ in "__main__":
    path = Path(__file__).parent
    outpath = path / "fit" 

    # make sure outpath exists
    if not outpath.exists():
        outpath.mkdir()

    group = parse_args()

    data = pd.read_csv(path / "data" / group / f"data_{group}_all_subjects.csv")

    with open(path.parent / "models" /"hierachical_IGT_ORL_paper.stan") as f:
        model_spec = f.read()
    

    summary = fit_group_level_one_group(
            data = data,
            model_spec = model_spec,
            savepath = outpath / f"param_est_{group}.csv",
            summary = True
        )

    # save summary
    summary.to_csv(outpath / f"param_est_{group}_summary.csv")
