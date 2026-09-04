import pandas as pd
import matplotlib.pyplot as plt
try:
    df = pd.read_csv("fairness_baseline.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
except FileNotFoundError:
    print("데이터 파일이 없습니다. 먼저 분석을 진행하세요!")
    exit()

fig, ax1 = plt.subplots(figsize=(12, 6))

# 1. Total Throughput 그래프 (막대 그래프)
ax1.set_xlabel('Time (HH:MM:SS)')
ax1.set_ylabel('Total Throughput (Mbps)', color='tab:blue', fontsize=12)
bars = ax1.bar(df['timestamp'], df['total_dl_mbps'], color='skyblue', alpha=0.6, width=0.0001, label='Throughput')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# 2. Fairness Index 그래프 (꺾은선 그래프)
ax2 = ax1.twinx()
ax2.set_ylabel("Jain's Fairness Index", color='tab:red', fontsize=12)
line = ax2.plot(df['timestamp'], df['fairness'], color='tab:red', marker='o', linewidth=2, label='Fairness Index')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.1)  # Fairness는 0~1 사이값

# 제목 및 범례
plt.title('5G Multi-UE Performance Analysis: Throughput vs Fairness', fontsize=14)
fig.tight_layout()

# 그래프 저장
plt.savefig('kpi_analysis_plot.png')
print("그래프가 'kpi_analysis_plot.png'로 저장되었습니다.")
plt.show()
