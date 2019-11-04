###### Python setup ######

if [[ ! -f /.dockerenv ]]; then
	source ${HOME}/.anaconda3/etc/profile.d/conda.sh
	conda activate chainer5
fi

if [[ $GDB == "1" ]]; then
	PYTHON="gdb -ex run --args python"

elif [[ $PROFILE == "1" ]]; then
	PYTHON="python -m cProfile -o profile"

else
	PYTHON="python"

fi

SCRIPT="../main.py"
DATA=${DATA:-/home/korsch/Data/info.yml}

###### Configurations ######

GPU=${GPU:-0}
N_JOBS=${N_JOBS:-0}
INPUT_SIZE=${INPUT_SIZE:-427}
BATCH_SIZE=${BATCH_SIZE:-16}
THRESH_TYPE=${THRESH_TYPE:-otsu}

MODEL_TYPE=${MODEL_TYPE:-inception}
PREPARE_TYPE=${PREPARE_TYPE:-model}

DATASET=${DATASET:-CUB200}
LABEL_SHIFT=${LABEL_SHIFT:-1}

N_PARTS=${N_PARTS:-4}

###### Argument construction ######

OPTS="${OPTS} --gpu $GPU"
OPTS="${OPTS} --K $N_PARTS"
OPTS="${OPTS} --n_jobs $N_JOBS"
OPTS="${OPTS} --model_type $MODEL_TYPE"
OPTS="${OPTS} --batch_size $BATCH_SIZE"
OPTS="${OPTS} --input_size $INPUT_SIZE"
OPTS="${OPTS} --label_shift $LABEL_SHIFT"
OPTS="${OPTS} --prepare_type $PREPARE_TYPE"
OPTS="${OPTS} --thresh_type $THRESH_TYPE"
OPTS="${OPTS} --no_center_crop_on_val"

export OMP_NUM_THREADS=2
