EXP_PATH=$1;
SCRIPT_PATH=$2;

PYTHON_PATH=${SCRIPT_PATH%%/}/do_experiment.py

cd ${EXP_PATH}
N_VAL=0
for d in ./experiment_*/; do
    python ${PYTHON_PATH} --exp_path ${EXP_PATH} --n_val ${N_VAL}
    ((N_VAL++))
done
