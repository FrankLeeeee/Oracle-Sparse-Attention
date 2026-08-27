#!/bin/bash
until grep -q SF_FINAL_DONE /data/projects/vision-gen/sglang/results/investigation/sparse_baselines/sf_final.log && grep -q CF_FINAL_DONE /data/projects/vision-gen/sglang/results/investigation/sparse_baselines/cf_final.log; do sleep 30; done
bash retime_serial.sh > /data/projects/vision-gen/sglang/results/investigation/sparse_baselines/retime.log 2>&1
