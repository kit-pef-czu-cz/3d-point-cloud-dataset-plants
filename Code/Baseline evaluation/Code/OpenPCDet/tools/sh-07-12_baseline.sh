#!/bin/bash
./sh-train-nested.sh ./cfgs/base_pointrcnn_mdl.yaml ./results/baseline_pointrcnn 150 repo_rnd_0 30 15 ap_sum_0.3 max
./sh-train-nested.sh ./cfgs/base_second_mdl.yaml ./results/baseline_second 150 repo_rnd_0 30 15 ap_sum_0.3 max
sudo shutdown -h now
