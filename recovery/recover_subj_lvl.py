import stan
from pathlib import Path
import numpy as np
import pandas as pd

# local imports 
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.fit_model import fit_subject_level
from utils.helper_functions import parse_n_subj



if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fit" / "subj_lvl"

    if not outpath.exists():
        outpath.mkdir(parents = True)

    with open(path.parent / "IGT_ORL.stan") as f:
        model_spec = f.read()

    # load the simulated data
    n_subs = parse_n_subj()
    sim_path = path / "simulated" / "subj_lvl" / f"ORL_{n_subs}_sub.csv"

    data = pd.read_csv(sim_path)

    for sub in range(1, n_subs + 1):
        tmp_data = data[data["sub"]==sub]
    
        fit_subject_level(
            data = tmp_data,
            model_spec = model_spec,
            savepath = outpath / f"param_rec_subj_{sub}.csv"
            )