#!/bin/bash
cd ~/study/monetlab/openairinterface5g/cmake_targets/ran_build/build
sudo ./nr-softmodem \
  -O /home/eunjeong/study/monetlab/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.e2.ej.conf \
  --rfsim 2>&1 | tee /home/eunjeong/study/monetlab/gnb_live.log
