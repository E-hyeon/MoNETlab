"""
collect_kpi_fast.py — nrMAC_stats.log(1초마다 gNB가 덮어쓰는 통계 파일)를
1초 간격으로 직접 폴링해서 kpi_fast.csv에 기록.

gnb_live.log(stdout)의 UE stats 블록은 시뮬레이션 프레임 카운트 기반 트리거라
(느리게 도는 소프트모뎀 탓에) 실제로는 5~8초에 한 번만 찍힌다.
반면 nrMAC_stats.log는 nrmac_stats_thread()가 sleep(1)로 매 1초마다
덮어쓰므로, 이걸 직접 폴링하면 훨씬 촘촘한 시계열을 얻을 수 있다.
"""

import re
import csv
import time
import sys
from datetime import datetime

STATS_LOG      = '/home/eunjeong/study/monetlab/openairinterface5g/cmake_targets/ran_build/build/nrMAC_stats.log'
OUTPUT         = 'kpi_fast.csv'
POLL_INTERVAL  = 1.0
FIELDS         = ["timestamp", "rnti", "dl_bytes", "ul_bytes", "mcs_ul", "nprb", "snr", "bler"]

LCID4_RE  = re.compile(r'UE (\w+):.*LCID 4: TX\s+(\d+) RX\s+(\d+)')
ULSCH_RE  = re.compile(r'UE (\w+): ulsch.*BLER (\S+) MCS \(\d+\) (\d+).*NPRB\s+(\d+).*SNR (\S+) dB')

prev_dl = {}


def main():
    with open(OUTPUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    print(f"[collect_kpi_fast] {STATS_LOG} 를 {POLL_INTERVAL}s 간격으로 폴링 -> {OUTPUT}")
    n_written = 0
    try:
        while True:
            ts = datetime.now().isoformat()
            try:
                with open(STATS_LOG) as f:
                    content = f.read()
            except (FileNotFoundError, OSError):
                time.sleep(POLL_INTERVAL)
                continue

            ue_data = {}
            for line in content.splitlines():
                m = LCID4_RE.search(line)
                if m:
                    rnti, tx, rx = m.group(1), int(m.group(2)), int(m.group(3))
                    ue_data.setdefault(rnti, {})
                    dl_diff = rx - prev_dl.get(rnti, rx)
                    prev_dl[rnti] = rx
                    ue_data[rnti]['dl_bytes'] = dl_diff
                    ue_data[rnti]['ul_bytes'] = tx

                m = ULSCH_RE.search(line)
                if m:
                    rnti = m.group(1)
                    ue_data.setdefault(rnti, {}).update({
                        'rnti':   rnti,
                        'bler':   m.group(2),
                        'mcs_ul': m.group(3),
                        'nprb':   m.group(4),
                        'snr':    m.group(5),
                    })

            if ue_data:
                with open(OUTPUT, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    for rnti, d in ue_data.items():
                        if 'mcs_ul' in d and 'dl_bytes' in d:
                            writer.writerow({
                                'timestamp': ts,
                                'rnti':      rnti,
                                'dl_bytes':  d.get('dl_bytes', 0),
                                'ul_bytes':  d.get('ul_bytes', 0),
                                'mcs_ul':    d.get('mcs_ul', ''),
                                'nprb':      d.get('nprb', ''),
                                'snr':       d.get('snr', ''),
                                'bler':      d.get('bler', ''),
                            })
                            n_written += 1

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n[collect_kpi_fast] 종료 (총 {n_written}행 기록)")


if __name__ == '__main__':
    main()
