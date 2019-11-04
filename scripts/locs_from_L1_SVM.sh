#!/usr/bin/env bash

source 00_config.sh

if [[ -z $WEIGHTS ]]; then
	echo "WEIGHTS variable is missing!"
	exit -1
fi

OPTS="${OPTS} --weights ${WEIGHTS}"
# OPTS="${OPTS} --visualize_coefs"
# OPTS="${OPTS} --scale_features"

SVM_OUTPUT=${SVM_OUTPUT:-"../.out"}
TRAINED_SVM="${SVM_OUTPUT}/clf_svm_${DATASET}_GLOBAL.${MODEL_TYPE}_glob_only_sparse_coefs.npz"

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${TRAINED_SVM} \
	${OPTS} \
	$@
