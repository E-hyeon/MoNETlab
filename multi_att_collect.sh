#!/bin/bash
# 여러 att 조건에서 KPI 수집 → kpi_combined.csv 생성
# gNB 재시작은 매번 수동으로 해야 함

cd ~/study/monetlab

CONF="openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.e2.ej.conf"
COMBINED="kpi_combined.csv"
COLLECT_SEC=180  # 조건당 3분 수집

# att=12 데이터는 이미 수집됨 → 기존 파일로 시작
echo "[초기화] att=12 기존 데이터 복사..."
cp kpi_baseline.csv "$COMBINED"
ROWS=$(tail -n +2 "$COMBINED" | wc -l)
echo "  att=12: $ROWS 행 보존"

for ATT in 25 35 45; do
    echo ""
    echo "========================================="
    echo "ATT = ${ATT} dB 로 변경 중..."

    python3 - <<PYEOF
import re
with open("$CONF") as f:
    txt = f.read()
txt = re.sub(r'att_tx\s*=\s*\d+', 'att_tx         = $ATT', txt)
txt = re.sub(r'att_rx\s*=\s*\d+', 'att_rx         = $ATT', txt)
with open("$CONF", "w") as f:
    f.write(txt)
print("  gnb.conf 업데이트: att_tx=$ATT att_rx=$ATT")
PYEOF

    echo ""
    echo ">>> gNB를 재시작하세요 (기존 gNB 터미널에서 Ctrl+C 후 동일 명령 재실행)"
    echo ">>> UE 두 개가 모두 'Registration Accept' 뜨면 ENTER..."
    read

    echo "iperf3 + KPI 컬렉터 재시작..."
    bash full_restart.sh

    echo "수집 중 (${COLLECT_SEC}초)..."
    sleep $COLLECT_SEC

    ROWS=$(tail -n +2 kpi_baseline.csv | wc -l)
    tail -n +2 kpi_baseline.csv >> "$COMBINED"
    echo "  att=${ATT}: ${ROWS} 행 추가"
done

# config를 att=12 로 복원
python3 - <<PYEOF
import re
with open("$CONF") as f:
    txt = f.read()
txt = re.sub(r'att_tx\s*=\s*\d+', 'att_tx         = 12', txt)
txt = re.sub(r'att_rx\s*=\s*\d+', 'att_rx         = 12', txt)
with open("$CONF", "w") as f:
    f.write(txt)
print("gnb.conf att 값 12로 복원 완료")
PYEOF

echo ""
TOTAL=$(tail -n +2 "$COMBINED" | wc -l)
echo "완료! kpi_combined.csv — 총 ${TOTAL} 행"
echo "다음 단계: python3 chronos_train.py"
