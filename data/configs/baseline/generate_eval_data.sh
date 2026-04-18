#!/usr/bin/env bash

# Check if a baseline argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 ae | vae_rnn | transformer"
  exit 1
fi

baseline="$1"
inds=(I9 I8 I7 I6 I5)
noises=(n0 n5 n10 n20 n50)

for ind in "${inds[@]}"; do
  for n in "${noises[@]}"; do
    echo "Generating eval data for baseline ${baseline} ${ind} ${n}"
    python -m data.mat2tensor \
      --mode pinpoint_eval \
      --config "data/configs/baseline/${baseline}/${ind}/${n}.yaml"
  done
done
