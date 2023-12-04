import stan
from pathlib import Path
import numpy as np
import pandas as pd


def recover(data, model_spec, savepath = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    
    model_spec : str
        Stan model specification.
    savepath : Path, optional
        Path to save the fitted parameters to. The default is None.
    """

    data_dict = {
        "choice" : np.array(data["choice"]).astype(int),
        "outcome" : np.array(data["outcome"]),
        "sign_out" : np.array(data["sign_out"]),
        "T": int(len(data)), #total number of trials
    }

    # fit the model
    model = stan.build(model_spec, data = data_dict)
    fit = model.sample(
        num_chains = 4, 
        num_samples = 2000,
        num_warmup = 1000)

    # get the estimated parameters
    df = fit.to_frame()

    # save the data
    if savepath:
        df.to_csv(savepath, index = False)



if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fit" / "subj_lvl"

    if not outpath.exists():
        outpath.mkdir(parents = True)

    with open(path.parent / "IGT_ORL.stan") as f:
        model_spec = f.read()

    # load the simulated data
    n_subs = 100
    sim_path = path / "simulated" / "subj_lvl" / f"ORL_{n_subs}_sub.csv"

    data = pd.read_csv(sim_path)

    for sub in range(1, n_subs + 1):
        tmp_data = data[data["sub"]==sub]
    
        recover(
            data = tmp_data,
            model_spec = model_spec,
            savepath = outpath / f"param_rec_subj_{sub}.csv"
            )