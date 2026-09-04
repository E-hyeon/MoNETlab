"""
MoNETlab 랩미팅 PPT 생성 스크립트
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm
import copy

# ── 색상 팔레트 ──────────────────────────────────────────────────
NAVY    = RGBColor(0x1A, 0x3A, 0x5C)   # 타이틀 배경
BLUE    = RGBColor(0x1F, 0x6F, 0xEB)   # 강조 색
LIGHT   = RGBColor(0xEF, 0xF4, 0xFF)   # 연한 배경
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x1A, 0x2E)
GRAY    = RGBColor(0x55, 0x65, 0x7A)
GREEN   = RGBColor(0x10, 0xB9, 0x81)
ORANGE  = RGBColor(0xF5, 0x9E, 0x0B)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]  # 완전 빈 레이아웃


# ── 헬퍼 ─────────────────────────────────────────────────────────
def add_rect(slide, l, t, w, h, fill=None, line=None, line_w=None):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.line.fill.background()
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        if line_w:
            shape.line.width = line_w
    return shape


def add_text(slide, text, l, t, w, h,
             size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT,
             wrap=True, italic=False):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txb


def header_bar(slide, title, subtitle=None):
    """슬라이드 상단 네이비 헤더 + 파란 줄"""
    add_rect(slide, 0, 0, 13.33, 1.1, fill=NAVY)
    add_rect(slide, 0, 1.1, 13.33, 0.07, fill=BLUE)
    add_text(slide, title, 0.35, 0.18, 10, 0.75,
             size=28, bold=True, color=WHITE)
    if subtitle:
        add_text(slide, subtitle, 0.35, 0.78, 12, 0.35,
                 size=13, color=RGBColor(0xB0, 0xC8, 0xFF), italic=True)


def bullet_box(slide, lines, l, t, w, h,
               title=None, title_color=BLUE, bullet="▸"):
    if title:
        add_text(slide, title, l, t, w, 0.38,
                 size=14, bold=True, color=title_color)
        t += 0.38
        h -= 0.38
    add_rect(slide, l, t, w, h, fill=LIGHT,
             line=RGBColor(0xC8, 0xD8, 0xF0), line_w=9000)
    for i, line in enumerate(lines):
        add_text(slide, f"{bullet}  {line}", l+0.15, t+0.1+i*0.35, w-0.2, 0.35,
                 size=13, color=DARK)


def code_box(slide, code_lines, l, t, w, h):
    add_rect(slide, l, t, w, h, fill=RGBColor(0x1E, 0x1E, 0x2E))
    for i, line in enumerate(code_lines):
        add_text(slide, line, l+0.12, t+0.1+i*0.28, w-0.15, 0.28,
                 size=10.5, color=RGBColor(0xA8, 0xE6, 0xCF),
                 bold=False)


def tag(slide, text, l, t, fill=BLUE, text_color=WHITE, size=11):
    w = len(text) * 0.085 + 0.25
    add_rect(slide, l, t, w, 0.29, fill=fill)
    add_text(slide, text, l+0.07, t+0.03, w-0.05, 0.26,
             size=size, bold=True, color=text_color, align=PP_ALIGN.CENTER)
    return l + w + 0.1


# ══════════════════════════════════════════════════════════════════
# Slide 1: Title
# ══════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(BLANK)
add_rect(s1, 0, 0, 13.33, 7.5, fill=NAVY)
add_rect(s1, 0, 5.8, 13.33, 1.7, fill=RGBColor(0x0F, 0x25, 0x40))
add_rect(s1, 0.4, 2.3, 0.08, 2.5, fill=BLUE)  # 세로 파란 줄

add_text(s1, "MoNETlab Weekly Progress", 0.65, 2.0, 12, 0.7,
         size=18, color=RGBColor(0x80, 0xB3, 0xFF), italic=True)
add_text(s1, "O-RAN AI 기반 RAN 스케줄링\ndApp 개발 현황", 0.65, 2.6, 12, 1.6,
         size=36, bold=True, color=WHITE)
add_text(s1, "AI-based PRB Scheduling with Channel KPI Prediction", 0.65, 4.25, 12, 0.5,
         size=15, color=RGBColor(0xB0, 0xC8, 0xFF))

add_text(s1, "2026년 6월  |  이은정  |  Catholic University", 0.65, 6.05, 12, 0.45,
         size=13, color=RGBColor(0x70, 0x90, 0xB0))

# 오른쪽 아이콘 박스
add_rect(s1, 9.8, 1.8, 3.1, 3.6, fill=RGBColor(0x12, 0x2A, 0x45),
         line=BLUE, line_w=18000)
for i, (lbl, val) in enumerate([
    ("UE 수", "2개 (UE1/UE2)"),
    ("KPI 피처", "6개"),
    ("모델", "TCN / LSTM"),
    ("목표", "Fairness + 예측"),
]):
    add_text(s1, lbl, 10.0, 2.1+i*0.78, 1.1, 0.35, size=10,
             color=RGBColor(0x80, 0xB3, 0xFF))
    add_text(s1, val, 10.0, 2.38+i*0.78, 2.7, 0.35, size=13,
             bold=True, color=WHITE)


# ══════════════════════════════════════════════════════════════════
# Slide 2: 목차
# ══════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(BLANK)
add_rect(s2, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s2, "목차 (Agenda)")

items = [
    ("01", "시스템 아키텍처", "OAI gNB + 5GC + RIC 구성 및 데이터 흐름"),
    ("02", "KPI 수집 모듈", "gnb_live.log 파싱 → CSV 저장"),
    ("03", "모델 비교", "Chronos TCN  vs  Global LSTM"),
    ("04", "dApp 컨트롤러", "예측 기반 공정성 가중치 산출 및 적용"),
    ("05", "성능 평가", "MAE / RMSE / Inference Latency / Fairness"),
    ("06", "향후 계획", "xApp 컨테이너화 및 E2 인터페이스 연동"),
]

colors = [BLUE, GREEN, ORANGE,
          RGBColor(0x8B,0x5C,0xF6), RGBColor(0xEC,0x48,0x99),
          RGBColor(0x14,0xB8,0xA6)]

for i, (num, title, desc) in enumerate(items):
    row, col = divmod(i, 2)
    lx = 0.5 + col * 6.4
    ty = 1.5 + row * 1.8
    add_rect(s2, lx, ty, 6.0, 1.55, fill=WHITE,
             line=colors[i], line_w=20000)
    add_rect(s2, lx, ty, 0.7, 1.55, fill=colors[i])
    add_text(s2, num, lx+0.05, ty+0.42, 0.65, 0.65,
             size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s2, title, lx+0.85, ty+0.2, 4.8, 0.45,
             size=16, bold=True, color=DARK)
    add_text(s2, desc, lx+0.85, ty+0.68, 4.8, 0.7,
             size=11, color=GRAY)


# ══════════════════════════════════════════════════════════════════
# Slide 3: 시스템 아키텍처
# ══════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(BLANK)
add_rect(s3, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s3, "시스템 아키텍처", "O-RAN Testbed Infrastructure")

# 왼쪽: 스택 다이어그램
layers = [
    (BLUE,   "Near-RT RIC (Kubernetes)",      "r4 Helm 릴리즈 | 192.168.70.x"),
    (GREEN,  "OAI gNB (rfsimulator)",          "Band 78 | 106 PRB | E2 Agent"),
    (ORANGE, "5G Core (Docker)",               "AMF/SMF/UPF | 192.168.70.0/24"),
    (RGBColor(0x8B,0x5C,0xF6), "UE1 / UE2 namespace", "oaitun_ue1/ue2 | iperf3 UDP 20Mbps"),
]
for i, (col, lbl, sub) in enumerate(layers):
    ty = 1.5 + i * 1.3
    add_rect(s3, 0.4, ty, 5.5, 1.1, fill=col)
    add_text(s3, lbl, 0.6, ty+0.08, 5.2, 0.45,
             size=14, bold=True, color=WHITE)
    add_text(s3, sub, 0.6, ty+0.52, 5.2, 0.45,
             size=11, color=RGBColor(0xDD,0xEE,0xFF))
    if i < len(layers)-1:
        add_text(s3, "▼", 2.8, ty+1.1, 0.6, 0.2,
                 size=16, bold=True, color=GRAY, align=PP_ALIGN.CENTER)

# 오른쪽: 데이터 흐름
add_text(s3, "데이터 흐름", 6.5, 1.35, 6.5, 0.4,
         size=14, bold=True, color=NAVY)
flow = [
    "gnb_live.log  (OAI gNB 실시간 출력)",
    "  ↓  collect_kpi.py  /  collect_kpi_live.py",
    "kpi_baseline.csv  /  kpi_live.csv",
    "  ↓  train_rnn_global.py  /  chronos_train.py",
    "model_global.pth  /  chronos_forecaster/",
    "  ↓  dapp_controller_global.py  /  _chronos.py",
    "/tmp/dapp_weights.json  (PRB 가중치)",
    "  ↓  O-RAN E2 → gNB 스케줄러",
]
code_box(s3, flow, 6.5, 1.75, 6.5, 5.35)

# KPI 피처 태그
add_text(s3, "KPI Features:", 0.4, 6.62, 2.0, 0.35, size=11, bold=True, color=GRAY)
lx = 2.1
for f in ["snr", "bler", "nprb", "mcs_ul", "ul_bytes", "dl_bytes"]:
    lx = tag(s3, f, lx, 6.62, fill=NAVY)


# ══════════════════════════════════════════════════════════════════
# Slide 4: KPI 수집 모듈
# ══════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(BLANK)
add_rect(s4, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s4, "KPI 수집 모듈", "collect_kpi.py / collect_kpi_live.py")

# 왼쪽: 파싱 방법
bullet_box(s4, [
    "OAI gNB 로그를 stdin으로 실시간 수신 (tail -f)",
    "LCID 4 라인 → dl_bytes / ul_bytes 추출 (diff 계산)",
    "ulsch 라인 → BLER / MCS / NPRB / SNR 추출",
    "Frame.Slot 경계마다 CSV에 한 행씩 기록",
    "UE별 RNTI 키로 데이터 집계",
], 0.4, 1.35, 6.2, 2.2, title="파싱 방식", title_color=BLUE)

# 오른쪽: CSV 스키마
bullet_box(s4, [
    "timestamp  — ISO8601 현재 시각",
    "rnti       — UE 식별자 (hex, e.g. 637b, c68b)",
    "dl_bytes   — 다운링크 증분 바이트",
    "ul_bytes   — 업링크 누적 바이트",
    "mcs_ul     — 업링크 MCS 인덱스",
    "nprb       — 할당 PRB 수",
    "snr        — 업링크 SNR (dB)",
    "bler       — 블록 에러율",
], 6.8, 1.35, 6.1, 3.1, title="CSV 스키마 (8개 컬럼)", title_color=GREEN)

# 데이터 현황
add_text(s4, "수집 데이터 현황", 0.4, 3.75, 12.5, 0.4,
         size=14, bold=True, color=NAVY)
stats = [
    ("kpi_baseline.csv", "3,508행", "2026-06-19 수집", NAVY),
    ("kpi_live.csv",     "622행",   "실시간 업데이트", GREEN),
    ("kpi_combined.csv", "1,708행", "baseline + live 병합", ORANGE),
]
for i, (name, rows, note, col) in enumerate(stats):
    lx = 0.4 + i * 4.2
    add_rect(s4, lx, 4.2, 3.9, 1.15, fill=col)
    add_text(s4, name, lx+0.15, 4.28, 3.6, 0.38,
             size=13, bold=True, color=WHITE)
    add_text(s4, rows, lx+0.15, 4.62, 3.6, 0.38,
             size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s4, note, lx+0.15, 4.98, 3.6, 0.3,
             size=10, color=RGBColor(0xDD,0xEE,0xFF))

# 실행 명령
code_box(s4, [
    "# 베이스라인 수집",
    "tail -f gnb_live.log | python3 collect_kpi.py",
    "",
    "# 라이브 수집",
    "tail -f gnb_live.log | python3 collect_kpi_live.py",
], 0.4, 5.55, 12.5, 1.72)


# ══════════════════════════════════════════════════════════════════
# Slide 5: 모델 비교
# ══════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(BLANK)
add_rect(s5, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s5, "모델 비교", "Chronos TCN  vs  Global Attention-LSTM")

# Chronos TCN 박스
add_rect(s5, 0.35, 1.3, 6.0, 5.9, fill=WHITE,
         line=BLUE, line_w=22000)
add_rect(s5, 0.35, 1.3, 6.0, 0.5, fill=BLUE)
add_text(s5, "BigDL Chronos TCNForecaster", 0.5, 1.34, 5.7, 0.42,
         size=14, bold=True, color=WHITE)

tcn_items = [
    ("아키텍처", "Temporal Convolutional Network (TCN)"),
    ("입력", "과거 10 스텝 (LOOKBACK=10)"),
    ("출력", "다음 1 스텝 예측 (HORIZON=1)"),
    ("학습 에포크", "300 (Early Stopping 내장)"),
    ("스케일러", "StandardScaler → 역정규화"),
    ("가속", "OpenVINO FP32 최적화 지원"),
    ("학습 스크립트", "chronos_train.py / chronos_final.py"),
    ("라이브 재학습", "chronos_live.py (kpi_live.csv 기반)"),
    ("저장 파일", "chronos_forecaster/ + scaler_chronos.pkl"),
]
for i, (k, v) in enumerate(tcn_items):
    ty = 1.9 + i * 0.55
    add_text(s5, k, 0.5, ty, 1.8, 0.45, size=11, bold=True, color=GRAY)
    add_text(s5, v, 2.3, ty, 3.8, 0.45, size=11, color=DARK)

# Global LSTM 박스
add_rect(s5, 6.95, 1.3, 6.0, 5.9, fill=WHITE,
         line=GREEN, line_w=22000)
add_rect(s5, 6.95, 1.3, 6.0, 0.5, fill=GREEN)
add_text(s5, "GlobalChannelRNN (LSTM + Attention)", 7.1, 1.34, 5.7, 0.42,
         size=14, bold=True, color=WHITE)

lstm_items = [
    ("아키텍처", "LSTM 2레이어 + Attention 헤드"),
    ("입력", "(B, T=10, 6) + UE 임베딩 (8d)"),
    ("출력", "다음 스텝 KPI (B, 6)"),
    ("LSTM 설정", "hidden=128, dropout=0.2"),
    ("UE 처리", "RNTI → LabelEncoder → 임베딩"),
    ("미지 UE", "padding_idx=n_ues → 제로 벡터"),
    ("학습 스크립트", "train_rnn_global.py"),
    ("라이브 재학습", "train_rnn_global_live.py (fine-tune)"),
    ("저장 파일", "model_global.pth + ue_encoder.pkl"),
]
for i, (k, v) in enumerate(lstm_items):
    ty = 1.9 + i * 0.55
    add_text(s5, k, 7.1, ty, 1.8, 0.45, size=11, bold=True, color=GRAY)
    add_text(s5, v, 8.9, ty, 3.8, 0.45, size=11, color=DARK)

add_text(s5, "vs", 6.16, 3.85, 0.6, 0.65,
         size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════
# Slide 6: dApp 컨트롤러
# ══════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(BLANK)
add_rect(s6, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s6, "dApp 컨트롤러", "dapp_controller_chronos.py / dapp_controller_global.py")

# 동작 흐름 (상단)
add_text(s6, "동작 흐름", 0.4, 1.28, 12.5, 0.38,
         size=14, bold=True, color=NAVY)
steps = [
    ("①", "KPI 읽기",   "kpi_live.csv\n최신 행 로드"),
    ("②", "히스토리 관리", "UE별 LOOKBACK=10\n슬라이딩 버퍼"),
    ("③", "예측",       "Chronos TCN\n또는 Global LSTM"),
    ("④", "가중치 계산", "Fairness 공식\n(1/SNR)×(1+BLER×5)"),
    ("⑤", "가중치 적용", "/tmp/dapp_weights.json\n1초 주기 갱신"),
]
colors5 = [BLUE, GREEN, ORANGE, RGBColor(0x8B,0x5C,0xF6), RGBColor(0xEC,0x48,0x99)]
for i, (num, title, desc) in enumerate(steps):
    lx = 0.35 + i * 2.55
    add_rect(s6, lx, 1.7, 2.35, 1.55, fill=colors5[i])
    add_text(s6, num,   lx+0.1, 1.75, 0.5, 0.4, size=18, bold=True, color=WHITE)
    add_text(s6, title, lx+0.1, 2.1,  2.1, 0.4, size=12, bold=True, color=WHITE)
    add_text(s6, desc,  lx+0.1, 2.5,  2.1, 0.65, size=10, color=RGBColor(0xDD,0xEE,0xFF))
    if i < len(steps)-1:
        add_text(s6, "→", lx+2.35, 2.22, 0.25, 0.4,
                 size=16, bold=True, color=GRAY, align=PP_ALIGN.CENTER)

# 공정성 가중치 공식
add_text(s6, "공정성 가중치 공식 (Fairness Weight Formula)", 0.4, 3.45, 12.5, 0.38,
         size=14, bold=True, color=NAVY)
code_box(s6, [
    "# 채널이 나쁠수록 (SNR ↓, BLER ↑) → 더 많은 PRB 할당",
    "score[ue] = (1 / snr) × (1 + bler × 5)",
    "weight[ue] = score[ue] / Σ score[all_ues]",
    "",
    "# 예측 초기 (hist < LOOKBACK=10): 균등 가중치 적용",
    "weight[ue] = 1.0 / n_ues",
], 0.4, 3.88, 12.5, 1.85)

# 출력 예시
bullet_box(s6, [
    '{"637b": 0.4823, "c68b": 0.5177}  →  /tmp/dapp_weights.json',
    "1초 주기로 갱신 → gNB 스케줄러가 읽어 PRB 할당 결정",
    "UE별 RNTI가 자동 탐지됨 (kpi_live.csv의 rnti 컬럼)",
], 0.4, 5.9, 12.5, 1.35, title="출력 예시 및 동작", title_color=ORANGE)


# ══════════════════════════════════════════════════════════════════
# Slide 7: 성능 평가
# ══════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(BLANK)
add_rect(s7, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s7, "성능 평가", "eval_dapp.py — Chronos TCN vs Persistence Baseline")

# 평가 방법
bullet_box(s7, [
    "비교 대상: Chronos TCN (dApp)  vs  Persistence (현재값 = 다음값, OAI 기준)",
    "평가 방식: UE별 슬라이딩 윈도우 (LOOKBACK=10) → 전체 구간 예측",
    "메트릭: MAE, RMSE (역정규화 후 실제 단위), Inference Latency",
], 0.4, 1.28, 12.5, 1.3, title="평가 설계")

# 메트릭 테이블 헤더
add_text(s7, "Feature별 MAE 비교 (예측값)", 0.4, 2.82, 12.5, 0.38,
         size=14, bold=True, color=NAVY)
headers = ["Feature", "MAE (TCN)", "MAE (Persistence)", "개선율", "RMSE (TCN)", "RMSE (Persistence)"]
widths  = [1.7, 1.6, 2.1, 1.1, 1.6, 2.1]
lxs     = [0.4]
for w in widths[:-1]:
    lxs.append(lxs[-1] + w)

add_rect(s7, 0.4, 3.22, 12.5, 0.42, fill=NAVY)
for i, (h, lx) in enumerate(zip(headers, lxs)):
    add_text(s7, h, lx+0.05, 3.25, widths[i]-0.05, 0.35,
             size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

rows_data = [
    ["snr",     "0.xxxx", "0.xxxx", "+x.x%", "0.xxxx", "0.xxxx"],
    ["bler",    "0.xxxx", "0.xxxx", "+x.x%", "0.xxxx", "0.xxxx"],
    ["nprb",    "0.xxxx", "0.xxxx", "+x.x%", "0.xxxx", "0.xxxx"],
    ["mcs_ul",  "0.xxxx", "0.xxxx", "+x.x%", "0.xxxx", "0.xxxx"],
    ["ul_bytes","0.xxxx", "0.xxxx", "+x.x%", "0.xxxx", "0.xxxx"],
    ["dl_bytes","0.xxxx", "0.xxxx", "+x.x%", "0.xxxx", "0.xxxx"],
    ["평균",    "—",      "—",      "+x.x%", "—",      "—"],
]
for r, row in enumerate(rows_data):
    ty = 3.64 + r * 0.4
    bg = RGBColor(0xF0,0xF5,0xFF) if r % 2 == 0 else WHITE
    if r == len(rows_data)-1:
        bg = RGBColor(0xE8,0xFF,0xF0)
    add_rect(s7, 0.4, ty, 12.5, 0.38, fill=bg)
    for i, (cell, lx) in enumerate(zip(row, lxs)):
        col = GREEN if (i == 3 and r < len(rows_data)-1) else DARK
        if r == len(rows_data)-1:
            col = RGBColor(0x06,0x7A,0x4A)
        add_text(s7, cell, lx+0.05, ty+0.04, widths[i]-0.05, 0.3,
                 size=11, bold=(r==len(rows_data)-1), color=col,
                 align=PP_ALIGN.CENTER)

add_text(s7, "※ eval_dapp.py 실행 후 실측값으로 채워야 합니다.", 0.4, 6.6, 10, 0.35,
         size=10, italic=True, color=GRAY)

# Latency
add_rect(s7, 10.15, 6.15, 2.9, 1.1, fill=BLUE)
add_text(s7, "Inference Latency", 10.3, 6.18, 2.7, 0.35,
         size=11, bold=True, color=WHITE)
add_text(s7, "avg / P95 / P99 (ms)", 10.3, 6.52, 2.7, 0.35,
         size=10, color=RGBColor(0xDD,0xEE,0xFF))
add_text(s7, "→ eval_dapp.py 결과", 10.3, 6.85, 2.7, 0.3,
         size=9, italic=True, color=RGBColor(0xAA,0xCC,0xFF))


# ══════════════════════════════════════════════════════════════════
# Slide 8: Fairness 분석
# ══════════════════════════════════════════════════════════════════
s8 = prs.slides.add_slide(BLANK)
add_rect(s8, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s8, "Fairness 분석", "fairness.py — Jain's Fairness Index")

# 왼쪽: 공식 설명
add_text(s8, "Jain's Fairness Index", 0.4, 1.28, 6.0, 0.4,
         size=14, bold=True, color=NAVY)
code_box(s8, [
    "# Jain's Fairness Index 공식",
    "JFI = (Σ x_i)²  /  (n × Σ x_i²)",
    "",
    "# 값의 의미",
    "JFI = 1.0  →  완전 공정 (모든 UE 동등)",
    "JFI = 1/n  →  완전 불공정 (한 UE 독점)",
    "",
    "# fairness.py 적용 방식",
    "10초 구간별 UE nPRB 합계 → JFI 계산",
], 0.4, 1.72, 6.0, 2.95)

# 오른쪽: 구현 요약
bullet_box(s8, [
    "kpi_live.csv 로드 → 10초 단위 Grouper 집계",
    "UE별 nPRB 합계로 Jain's Index 계산",
    "평균 JFI / UE1 nPRB / UE2 nPRB 출력",
    "plot_fairness.py → kpi_analysis_plot.png 시각화",
], 6.8, 1.28, 6.0, 2.7, title="fairness.py 동작", title_color=GREEN)

# 기대 결과
add_text(s8, "기대 결과 (dApp 적용 전/후)", 0.4, 4.85, 12.5, 0.4,
         size=14, bold=True, color=NAVY)

cases = [
    ("dApp 미적용\n(OAI 기본 스케줄러)", "JFI ≈ 0.5~0.7\n채널 상태 무관 균등 배분 시도\n→ 실제론 채널 좋은 UE 집중",
     RGBColor(0xFE,0xE2,0xE2), RGBColor(0xEC,0x48,0x99)),
    ("dApp 적용\n(Chronos TCN 예측)",    "JFI ≈ 0.85~1.0\n채널 상태 예측 → 불량 UE 우선\n→ 공정한 PRB 배분",
     RGBColor(0xD1,0xFA,0xE5), GREEN),
]
for i, (title, desc, bg, col) in enumerate(cases):
    lx = 0.4 + i * 6.4
    add_rect(s8, lx, 5.28, 6.0, 1.95, fill=bg, line=col, line_w=18000)
    add_text(s8, title, lx+0.2, 5.35, 5.5, 0.55,
             size=13, bold=True, color=col)
    add_text(s8, desc, lx+0.2, 5.9, 5.5, 1.25, size=12, color=DARK)


# ══════════════════════════════════════════════════════════════════
# Slide 9: 향후 계획
# ══════════════════════════════════════════════════════════════════
s9 = prs.slides.add_slide(BLANK)
add_rect(s9, 0, 0, 13.33, 7.5, fill=RGBColor(0xF8, 0xFA, 0xFF))
header_bar(s9, "향후 계획", "Next Steps")

plans = [
    ("단기\n(1~2주)", [
        "eval_dapp.py 실행 → MAE/RMSE 실측 수치 확보",
        "train_rnn_global.py 구현 완료 (현재 파일 비어 있음)",
        "dapp_controller_global.py 구현 완료",
        "Fairness 실험: dApp 적용 전/후 JFI 비교",
    ], BLUE),
    ("중기\n(3~4주)", [
        "oai-channel-prediction/ xApp 컨테이너화",
        "E2 인터페이스 연동 (E2SM-KPM via Near-RT RIC)",
        "OpenVINO 가속 검증 (추론 latency 최적화)",
        "multi_att_collect.sh → 다중 채널 환경 실험",
    ], ORANGE),
    ("장기", [
        "O-RAN Alliance 규격 준수 xApp 패키징",
        "Kubernetes Helm Chart 배포 자동화",
        "논문 작성: AI-based Fairness Scheduling in O-RAN",
        "공개 데이터셋 및 코드 릴리즈",
    ], GREEN),
]
for i, (period, items, col) in enumerate(plans):
    lx = 0.35 + i * 4.32
    add_rect(s9, lx, 1.28, 4.1, 0.55, fill=col)
    add_text(s9, period, lx+0.1, 1.32, 3.9, 0.47,
             size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s9, lx, 1.83, 4.1, 5.45, fill=WHITE,
             line=col, line_w=18000)
    for j, item in enumerate(items):
        add_text(s9, f"✓  {item}", lx+0.15, 1.95+j*1.1, 3.8, 0.95,
                 size=12, color=DARK)


# ══════════════════════════════════════════════════════════════════
# Slide 10: 마무리
# ══════════════════════════════════════════════════════════════════
s10 = prs.slides.add_slide(BLANK)
add_rect(s10, 0, 0, 13.33, 7.5, fill=NAVY)
add_rect(s10, 0, 3.5, 13.33, 0.08, fill=BLUE)
add_rect(s10, 0.4, 1.5, 0.09, 2.0, fill=BLUE)

add_text(s10, "감사합니다", 0.7, 1.6, 12, 1.0,
         size=48, bold=True, color=WHITE)
add_text(s10, "Questions & Discussion", 0.7, 2.65, 12, 0.55,
         size=20, color=RGBColor(0x80,0xB3,0xFF), italic=True)

add_text(s10, "이은정  |  rlojeong@catholic.ac.kr", 0.7, 3.85, 8, 0.45,
         size=14, color=RGBColor(0xB0,0xC8,0xFF))
add_text(s10, "Catholic University of Korea", 0.7, 4.28, 8, 0.38,
         size=12, color=GRAY)

# 요약 카드
add_rect(s10, 7.5, 3.75, 5.5, 3.4, fill=RGBColor(0x12,0x2A,0x45),
         line=BLUE, line_w=18000)
add_text(s10, "이번 주 완료 사항", 7.7, 3.82, 5.2, 0.38,
         size=13, bold=True, color=BLUE)
summary = [
    "✓ KPI 수집 파이프라인 구축 (3,508행)",
    "✓ Chronos TCN 학습 및 저장 완료",
    "✓ dApp 컨트롤러 Chronos 버전 구현",
    "✓ Fairness 분석 스크립트 (fairness.py)",
    "✓ 성능 평가 스크립트 (eval_dapp.py)",
    "✓ 라이브 재학습 파이프라인 구성",
]
for i, s in enumerate(summary):
    add_text(s10, s, 7.7, 4.25+i*0.47, 5.0, 0.42,
             size=11, color=WHITE)


# ── 저장 ─────────────────────────────────────────────────────────
out = '/home/eunjeong/study/monetlab/monetlab_weekly_progress.pptx'
prs.save(out)
print(f"저장 완료: {out}")
