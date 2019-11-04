#!/usr/bin/env bash

source 00_config.sh

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${TRAINED_SVM} \
	${OPTS} \
	$@
