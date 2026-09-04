import re, csv, sys
from datetime import datetime

output_file = "kpi_live.csv"
fields = ["timestamp","rnti","dl_bytes","ul_bytes","mcs_ul","nprb","snr","bler"]

with open(output_file, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fields).writeheader()

prev_dl = {}  # RNTI별 이전 dl_bytes (diff 계산용)
ue_data = {}

for line in sys.stdin:
    line = line.strip()
    ts = datetime.now().isoformat()

    # LCID 4: 누적 TX/RX bytes
    m = re.search(r'UE (\w+):.*LCID 4: TX\s+(\d+) RX\s+(\d+)', line)
    if m:
        rnti, tx, rx = m.group(1), int(m.group(2)), int(m.group(3))
        ue_data.setdefault(rnti, {})
        # diff 계산
        dl_diff = rx - prev_dl.get(rnti, rx)
        prev_dl[rnti] = rx
        ue_data[rnti]["dl_bytes"] = dl_diff
        ue_data[rnti]["ul_bytes"] = tx

    # UL 물리 계층: MCS, NPRB, SNR, BLER
    m = re.search(r'UE (\w+): ulsch.*BLER (\S+) MCS \(0\) (\d+).*NPRB\s+(\d+).*SNR (\S+) dB', line)
    if m:
        rnti = m.group(1)
        ue_data.setdefault(rnti, {}).update({
            "rnti": rnti,
            "bler": m.group(2),
            "mcs_ul": m.group(3),
            "nprb": m.group(4),
            "snr": m.group(5),
            "timestamp": ts,
        })

    # Frame.Slot마다 기록
    if "Frame.Slot" in line and ue_data:
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            for rnti, d in ue_data.items():
                if "mcs_ul" in d and "dl_bytes" in d:
                    writer.writerow({
                        "timestamp": d.get("timestamp", ts),
                        "rnti": rnti,
                        "dl_bytes": d.get("dl_bytes", 0),
                        "ul_bytes": d.get("ul_bytes", 0),
                        "mcs_ul":   d.get("mcs_ul", ""),
                        "nprb":     d.get("nprb", ""),
                        "snr":      d.get("snr", ""),
                        "bler":     d.get("bler", ""),
                    })
        ue_data = {}
