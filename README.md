# IGT_dementia
This repository holds the code used for the exam project for decision making (F2023) at Aarhus University. The project investigates decision-making in dementia patients compared to healthy controls using the Iowa Gambling Task (IGT) and the Outcome Representation Learning (ORL) model.

**NOTE:** WORK STILL IN PROGRESS

## Data
The data used for this project, was graciously provided by xxxx and xxxx. The data is not publicly available, and therefore not included in this repository.

## Pipeline
As the data is not available in this repository, the parameter estimation pipeline cannot be replicated. However, all needed files for the parameter estimation pipeline are available. To reproduce the results, follow the pipeline below.

Setup a virtual environment using the setup_env.sh script. This will install all needed packages.
```
bash setup_env.sh
```

### Parameter recovery
Activate the virtual environment
```
source env/bin/activate
```

Run the parameter recovery pipeline for subject level parameters
```
python recovery/simulate_subj_lvl.py
python recovery/recover_subj_lvl.py
python recovery/plot_subj_lvl.py
```

Run the parameter recovery for group level differences. 
```
python recovery/simulate_group_lvl.py
python recovery/recover_group_lvl.py
python recovery/plot_group_lvl.py
```
**Warning:** This will take a long time to run.

### Parameter estimation