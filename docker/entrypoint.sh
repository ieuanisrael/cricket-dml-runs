#!/bin/bash

set -eu

# Docker often runs without a TTY, so R block-buffers stdout and logs look "empty".
# script(1) runs the command with a pseudo-tty; -e propagates the child's exit status;
# -f flushes output; typescript is discarded to /dev/null.
exec script -qefc "Rscript /app/R/run_dml_caret_gpu.R \
  --data=data/raw/t20_ball_by_ball.csv \
  --gpu=TRUE \
  --folds=3 \
  --out=outputs/run_caret_cpu" /dev/null
