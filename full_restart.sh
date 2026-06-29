#!/bin/bash
set -e
cd ~/study/monetlab

echo "[1/6] ue2 tunnel rename..."
sudo ip netns exec ue2 ip link set oaitun_ue1 down 2>/dev/null || true
sudo ip netns exec ue2 ip link set oaitun_ue1 name oaitun_ue2 2>/dev/null || true
sudo ip netns exec ue2 ip link set oaitun_ue2 up 2>/dev/null || true

echo "[2/6] routing..."
# ue1: veth route (10.100.1.1) stays for rfsim, add data routes via oaitun_ue1
sudo ip netns exec ue1 ip route add 192.168.70.0/24 dev oaitun_ue1 2>/dev/null || true
sudo ip netns exec ue1 ip route add 10.0.0.0/8 dev oaitun_ue1 2>/dev/null || true
# ue2: same pattern with oaitun_ue2
sudo ip netns exec ue2 ip route add 192.168.70.0/24 dev oaitun_ue2 2>/dev/null || true
sudo ip netns exec ue2 ip route add 10.0.0.0/8 dev oaitun_ue2 2>/dev/null || true

echo "[3/6] iperf3 servers..."
docker exec oai-ext-dn pkill iperf3 2>/dev/null || true
sleep 2
docker exec -d oai-ext-dn iperf3 -s -B 192.168.70.135
docker exec -d oai-ext-dn iperf3 -s -B 192.168.70.135 -p 5202
sleep 2

echo "[4/6] UE IPs..."
UE1_IP=$(sudo ip netns exec ue1 ip addr show oaitun_ue1 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
UE2_IP=$(sudo ip netns exec ue2 ip addr show oaitun_ue2 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "    UE1=$UE1_IP UE2=$UE2_IP"

echo "[5/6] iperf3 clients..."
kill $(ps -ef | grep "iperf3 -c" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 1
sudo ip netns exec ue1 iperf3 -c 192.168.70.135 -B $UE1_IP -u -b 20M -t 3600 --logfile ~/study/monetlab/iperf_ue1.log &
sudo ip netns exec ue2 iperf3 -c 192.168.70.135 -B $UE2_IP -u -b 20M -p 5202 -t 3600 --logfile ~/study/monetlab/iperf_ue2.log &

echo "[6/6] KPI collector..."
kill $(ps -ef | grep collect_kpi | grep -v grep | awk '{print $2}') 2>/dev/null || true
rm -f ~/study/monetlab/kpi_baseline.csv
tail -f ~/study/monetlab/gnb_live.log | python3 ~/study/monetlab/collect_kpi.py &

echo "Done! 30초 후 확인하려면: sleep 30 && tail -3 ~/study/monetlab/iperf_ue1.log && tail -3 ~/study/monetlab/iperf_ue2.log && wc -l ~/study/monetlab/kpi_baseline.csv"
