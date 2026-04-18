#!/usr/bin/env bash

inds=(I9 I8 I7 I6 I5)
noises=(n0 n5 n10 n20 n50)

for ind in "${inds[@]}"; do
  for n in "${noises[@]}"; do
    echo "Generating eval data for PINPOINT ${ind} ${n}"
    python -m data.mat2tensor \
      --mode pinpoint_eval \
      --config "data/configs/pinpoint/eval/${ind}/${n}.yaml"
  done
done
