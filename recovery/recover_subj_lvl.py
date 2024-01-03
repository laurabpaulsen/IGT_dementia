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
    
    n_subs = parse_n_subj()

    outpath = path / "fit" / "subj_lvl"


    for theta_bool, folder in zip([True, False], [f"{n_subs}", f"{n_subs}_no_theta" ]):
        # make sure the output path exists
        (outpath / folder).mkdir(parents=True, exist_ok=True)

        sim_path = path / "simulated" / "subj_lvl" / f"{n_subs}"/ f"ORL.csv"

        data = pd.read_csv(sim_path)

        if theta_bool:
            model_path = path.parent / "models" / "IGT_ORL.stan"
        else:
            model_path = path.parent /  "models" / "IGT_ORL_no_theta.stan"

        with open(model_path, "r") as file:
            model_spec = file.read()

        for sub in range(1, n_subs + 1):
            tmp_data = data[data["sub"]==sub]
        
            fit_subject_level(
                data = tmp_data,
                model_spec = model_spec,
                savepath = outpath / folder / f"param_rec_subj_{sub}.csv"
                )
            print(f"Finished subject {sub} of {n_subs}")