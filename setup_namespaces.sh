#!/bin/bash
set -e

echo "[1/4] Cleaning up existing namespaces and veth pairs..."
sudo ip netns del ue1 2>/dev/null || true
sudo ip netns del ue2 2>/dev/null || true
sudo ip link del veth-ue1-host 2>/dev/null || true
sudo ip link del veth-ue2-host 2>/dev/null || true

echo "[2/4] Creating namespace ue1 with veth..."
sudo ip netns add ue1
sudo ip link add veth-ue1-host type veth peer name veth-ue1-ns
sudo ip link set veth-ue1-ns netns ue1
sudo ip addr add 10.100.1.1/30 dev veth-ue1-host
sudo ip link set veth-ue1-host up
sudo ip netns exec ue1 ip addr add 10.100.1.2/30 dev veth-ue1-ns
sudo ip netns exec ue1 ip link set veth-ue1-ns up
sudo ip netns exec ue1 ip link set lo up
sudo ip netns exec ue1 ip route add default via 10.100.1.1

echo "[3/4] Creating namespace ue2 with veth..."
sudo ip netns add ue2
sudo ip link add veth-ue2-host type veth peer name veth-ue2-ns
sudo ip link set veth-ue2-ns netns ue2
sudo ip addr add 10.100.2.1/30 dev veth-ue2-host
sudo ip link set veth-ue2-host up
sudo ip netns exec ue2 ip addr add 10.100.2.2/30 dev veth-ue2-ns
sudo ip netns exec ue2 ip link set veth-ue2-ns up
sudo ip netns exec ue2 ip link set lo up
sudo ip netns exec ue2 ip route add default via 10.100.2.1

echo "[4/4] Enabling IP forwarding..."
sudo sysctl -w net.ipv4.ip_forward=1 > /dev/null

echo ""
echo "Done! Now start UEs with:"
echo ""
echo "# UE1 (new terminal):"
echo "cd ~/study/monetlab/openairinterface5g/cmake_targets/ran_build/build"
echo "sudo ip netns exec ue1 ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim \\"
echo "  --rfsimulator.[0].serveraddr 10.100.1.1 \\"
echo "  --uecap_file ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/uecap_ports1.xml \\"
echo "  -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/ue1.conf \\"
echo "  2>&1 | tee ~/study/monetlab/ue1.log"
echo ""
echo "# UE2 (new terminal):"
echo "cd ~/study/monetlab/openairinterface5g/cmake_targets/ran_build/build"
echo "sudo ip netns exec ue2 ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim \\"
echo "  --rfsimulator.[0].serveraddr 10.100.2.1 \\"
echo "  --uecap_file ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/uecap_ports1.xml \\"
echo "  -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/ue2.conf \\"
echo "  2>&1 | tee ~/study/monetlab/ue2.log"
