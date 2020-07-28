#!/usr/bin/env bash

source 00_config.sh

PARTS=GLOBAL
$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${PARTS} \
	${TRAINED_SVM} \
	${OPTS} \
	$@
