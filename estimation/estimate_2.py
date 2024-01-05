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
    outpath = path / "fit" / "param_est_HC_AD_2.csv"
    summary_path = path / "fit" / "param_est_HC_AD_summary_2".csv"

    # load data
    AD_data = pd.read_csv(path / "data2" / "alzheimer_disease.csv")
    HC_data = pd.read_csv(path / "data2" / "healthy_controls.csv")
    HC_data["sub"] += AD_data["sub"].max() 

    AD_data["group"] = 1
    HC_data["group"] = 2

    data = pd.concat([AD_data, HC_data])

    with open(path.parent / "models" / "hierachical_IGT_ORL_compare.stan") as f:
        model_spec = f.read()
    
    summary = fit_group_level(
        data = data,
        model_spec = model_spec,
        savepath = outpath,
        summary = True
    )

    # save summary
    summary.to_csv(summary_path)
