#!/usr/bin/env bash

# OPTS="${OPTS} --visualize_coefs"

source config.sh

PREPARE_TYPE=${PREPARE_TYPE:-model}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
N_JOBS=${N_JOBS:-0}

OPTS=${OPTS:-""}
OPTS="${OPTS} --gpu $GPU"
OPTS="${OPTS} --batch_size $BATCH_SIZE"
OPTS="${OPTS} --n_jobs $N_JOBS"
OPTS="${OPTS} --prepare_type $PREPARE_TYPE"

# OPTS="${OPTS} --scale_features"


SVM_OUTPUT=${SVM_OUTPUT:-"../.out"}
TRAINED_SVM="${SVM_OUTPUT}/clf_svm_${DATASET}_GLOBAL.${MODEL_TYPE}_glob_only_sparse_coefs.npz"

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${TRAINED_SVM} \
	${OPTS} \
	$@
