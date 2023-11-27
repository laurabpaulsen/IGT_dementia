import stan
from pathlib import Path
import numpy as np
import pandas as pd





def recover_group_level(data, model_spec, savepath = None):
    """
    Generate synthetic data and fit the model to it. Check how well the parameters are recovered by plotting median against the true parameters.

    Parameters
    ----------
    n_groups : int
        Number of groups to simulate.
    model_spec : str
        Stan model specification.
    savepath : Path, optional
        Path to save the fitted parameters to. The default is None.
    """

    data_estimated = pd.DataFrame()


    # get the unique groups
    groups = data["group"].unique()

    for group in groups:
        # get the data for this group
        data_tmp = data[data["group"] == group]

        # fit the model
        model = stan.build(model_spec, data = data)
        fit = model.sample(num_chains = 4, num_samples = 1000)

        # get the estimated parameters
        df = fit.to_frame()

        # add the group number
        df["group"] = group
        
        # add to the data
        data_estimated = pd.concat([data_estimated, df])


    # save the data
    if savepath:
        data_estimated.to_csv(savepath, index = False)



if __name__ == "__main__":
    path = Path(__file__).parent

    outpath = path / "fit"

    if not outpath.exists():
        outpath.mkdir()

    with open(path.parent / "hierachical_IGT_ORL.stan") as f:
        model_spec = f.read()

    # load in the simulated data
    filename = "simulated_ORL_2_groups_3_sub.csv"
    data = pd.read_csv(path / "simulated" / filename)

    recover_group_level(
        data = data,
        model_spec = model_spec,
        savepath_df = outpath / f"param_rec_{filename}"
    )