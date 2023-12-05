import stan
from pathlib import Path
import numpy as np
import pandas as pd


# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.fit_model import fit_group_level

if __name__ in "__main__":
    path = Path(__file__).parent
    outpath = path / "fit" / "param_est_HC_AD.csv"
    summary_path = path / "fit" / "param_est_HC_AD_summary.csv"

    # load data
    AD_data = pd.read_csv(path / "data" / "AD" / "data_AD_all_subjects.csv")
    HC_data = pd.read_csv(path / "data" / "HC" / "data_HC_all_subjects.csv")
    HC_data["sub"] += AD_data["sub"].max() 

            
    AD_data["group"] = -0.5
    HC_data["group"] = 0.5

    data = pd.concat([AD_data, HC_data])

    with open(path.parent / "hierachical_IGT_ORL.stan") as f:
        model_spec = f.read()
    
    summary = fit_group_level(
        data = data,
        model_spec = model_spec,
        savepath = outpath,
        summary = True
    )

    # save summary
    summary.to_csv(summary_path)
