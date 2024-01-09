
# get location of this bash script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# activate environment one level up
source "${SCRIPT_DIR}/../env/bin/activate"

# run pipeline
python "${SCRIPT_DIR}/preprocess1.py"
python "${SCRIPT_DIR}/preprocess2.py"
python "${SCRIPT_DIR}/hierachical/estimate.py"
python "${SCRIPT_DIR}/hierachical/plot.py"
python "${SCRIPT_DIR}/hierachical/posterior_checks.py"

python "${SCRIPT_DIR}/hierachical_compare/estimate1.py"
python "${SCRIPT_DIR}/hierachical_compare/estimate2.py"
python "${SCRIPT_DIR}/hierachical_compare/estimate_pooled.py"
python "${SCRIPT_DIR}/hierachical_compare/plot.py"
python "${SCRIPT_DIR}/hierachical_compare/posterior_checks.py"

# deactivate environment
deactivate