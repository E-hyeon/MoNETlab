#!/bin/bash
# ue2 tunnel rename
sudo ip netns exec ue2 ip link set oaitun_ue1 down 2>/dev/null
sudo ip netns exec ue2 ip link set oaitun_ue1 name oaitun_ue2 2>/dev/null
sudo ip netns exec ue2 ip link set oaitun_ue2 up

# routing
sudo ip netns exec ue2 ip route add 192.168.70.0/24 dev oaitun_ue2 2>/dev/null
sudo ip netns exec ue1 ip route del default 2>/dev/null
sudo ip netns exec ue1 ip route add default dev oaitun_ue1
sudo ip netns exec ue2 ip route del default 2>/dev/null
sudo ip netns exec ue2 ip route add default dev oaitun_ue2

# iperf3 servers
docker exec -d oai-ext-dn iperf3 -s -B 192.168.70.135
docker exec -d oai-ext-dn iperf3 -s -B 192.168.70.135 -p 5202

echo "Done. Starting iperf3 clients..."
sudo ip netns exec ue1 iperf3 -c 192.168.70.135 -B 10.0.0.5 -u -b 20M -t 3600 --logfile iperf_ue1.log &
sudo ip netns exec ue2 iperf3 -c 192.168.70.135 -B 10.0.0.8 -u -b 20M -p 5202 -t 3600 --logfile iperf_ue2.log &
echo "All done!"
