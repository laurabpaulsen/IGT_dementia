
# set the number of groups to simulate
N_SUBJ=20
N_GROUPS=3

# get location of this bash script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# activate environment one level up
source "${SCRIPT_DIR}/../env/bin/activate"

# run pipeline
python "${SCRIPT_DIR}/simulate_group_lvl.py" --n_subj ${N_SUBJ} --n_groups ${N_GROUPS}
python "${SCRIPT_DIR}/recover_group_lvl.py" --n_subj ${N_SUBJ} --n_groups ${N_GROUPS}
python "${SCRIPT_DIR}/plot_group_lvl.py" --n_subj ${N_SUBJ} --n_groups ${N_GROUPS}

# deactivate environment
deactivate