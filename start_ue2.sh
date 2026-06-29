#!/bin/bash
cd ~/study/monetlab/openairinterface5g/cmake_targets/ran_build/build
sudo ip netns exec ue2 ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim --rfsimulator.[0].serveraddr 10.100.2.1 --uecap_file ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/uecap_ports1.xml -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/ue2.conf 2>&1 | tee ~/study/monetlab/ue2.log
