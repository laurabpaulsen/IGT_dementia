import stan
from pathlib import Path
import numpy as np
import pandas as pd


# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.fit_model import fit_group_level_one_group

if __name__ in "__main__":
    path = Path(__file__).parent
    outpath = path / "fit" 

    # load data
    AD_data = pd.read_csv(path / "data" / "AD" / "data_AD_all_subjects.csv")
    HC_data = pd.read_csv(path / "data" / "HC" / "data_HC_all_subjects.csv")
    

    with open(path.parent / "hierachical_IGT_ORL_one_group.stan") as f:
        model_spec = f.read()
    
    for data, group in zip([HC_data, AD_data], ["HC", "AD"]):

        tmp_outpath = outpath / f"param_est_{group}.csv"
        tmp_summarypath = outpath / f"param_est_{group}_summary.csv"

        summary = fit_group_level_one_group(
            data = data,
            model_spec = model_spec,
            savepath = tmp_outpath,
            summary = True
        )

        # save summary
        summary.to_csv(tmp_summarypath)
