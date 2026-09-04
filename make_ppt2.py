"""
Lab Meeting PPT - OAI DU/CU Channel Prediction Acceleration
실제 측정 수치 기반
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import datetime

TODAY = datetime.date.today().strftime("%Y%m%d")
OUT   = f"/home/eunjeong/study/monetlab/Lab_Meeting_Channel_Prediction_{TODAY}.pptx"

# ── 팔레트 ──────────────────────────────────────────────────────────
C_NAVY   = RGBColor(0x0D, 0x1B, 0x3E)
C_BLUE   = RGBColor(0x1A, 0x5C, 0xC8)
C_LBLUE  = RGBColor(0x3A, 0x86, 0xFF)
C_CYAN   = RGBColor(0x00, 0xB4, 0xD8)
C_GRAY   = RGBColor(0x4A, 0x5C, 0x6A)
C_LGRAY  = RGBColor(0xF0, 0xF4, 0xF8)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_DARK   = RGBColor(0x1A, 0x1A, 0x2E)
C_GREEN  = RGBColor(0x06, 0xD6, 0x8A)
C_ORANGE = RGBColor(0xFF, 0x8C, 0x00)
C_RED    = RGBColor(0xEF, 0x44, 0x44)
C_MINT   = RGBColor(0xE0, 0xF7, 0xF1)
C_BRED   = RGBColor(0xFE, 0xE2, 0xE2)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ── 공통 헬퍼 ────────────────────────────────────────────────────────
def rect(slide, l, t, w, h, fill=None, line_color=None, line_w=12700):
    s = slide.shapes.add_shape(1,
        Inches(l), Inches(t), Inches(w), Inches(h))
    s.line.fill.background()
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    else:
        s.fill.background()
    if line_color:
        s.line.color.rgb = line_color
        s.line.width = line_w
    return s

def txt(slide, text, l, t, w, h,
        size=14, bold=False, italic=False,
        color=C_DARK, align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(
        Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    p  = tf.paragraphs[0]; p.alignment = align
    r  = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold
    r.font.italic = italic; r.font.color.rgb = color
    return tb

def header(slide, title, sub=None):
    rect(slide, 0, 0, 13.33, 1.05, fill=C_NAVY)
    rect(slide, 0, 1.05, 13.33, 0.06, fill=C_LBLUE)
    txt(slide, title, 0.4, 0.12, 10, 0.6,
        size=26, bold=True, color=C_WHITE)
    if sub:
        txt(slide, sub, 0.4, 0.68, 12, 0.3,
            size=11, italic=True, color=RGBColor(0x9B,0xBC,0xFF))

def badge(slide, label, l, t, fill=C_BLUE, fc=C_WHITE, size=10):
    w = max(len(label)*0.088+0.22, 0.6)
    rect(slide, l, t, w, 0.27, fill=fill)
    txt(slide, label, l+0.06, t+0.03, w-0.08, 0.22,
        size=size, bold=True, color=fc, align=PP_ALIGN.CENTER)
    return l + w + 0.1

def code(slide, lines, l, t, w, h, bg=RGBColor(0x0D,0x17,0x27)):
    rect(slide, l, t, w, h, fill=bg)
    rect(slide, l, t, 0.04, h, fill=C_LBLUE)
    for i, ln in enumerate(lines):
        c = C_CYAN if ln.startswith("#") else RGBColor(0xA8,0xE6,0xCF)
        txt(slide, ln, l+0.15, t+0.1+i*0.275, w-0.2, 0.26,
            size=10, color=c)

def divider(slide, l, t, w):
    rect(slide, l, t, w, 0.03, fill=C_LBLUE)

def kv_row(slide, key, val, l, t, w, kw=2.2, bg=None, vc=C_DARK):
    if bg:
        rect(slide, l, t, w, 0.36, fill=bg)
    txt(slide, key, l+0.1, t+0.04, kw-0.1, 0.3, size=11, bold=True, color=C_GRAY)
    txt(slide, val, l+kw,  t+0.04, w-kw-0.1, 0.3, size=11, color=vc)


# ════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(BLANK)
rect(s1, 0, 0, 13.33, 7.5, fill=C_NAVY)
# 대각선 장식
rect(s1, 0, 5.5, 13.33, 2.0, fill=RGBColor(0x08,0x12,0x28))
rect(s1, 0, 5.48, 13.33, 0.05, fill=C_LBLUE)
rect(s1, 0.38, 1.9, 0.07, 2.8, fill=C_LBLUE)   # 세로선

txt(s1, "OAI DU/CU Channel Prediction Acceleration",
    0.6, 1.55, 11, 0.65, size=16,
    color=RGBColor(0x7B,0xB3,0xFF), italic=True)
txt(s1, "TX-based SNR Differentiation\n& dApp Integration",
    0.6, 2.1, 11, 1.4, size=38, bold=True, color=C_WHITE)
txt(s1, "AI-driven PRB Fairness Scheduling · O-RAN rfsimulator Testbed",
    0.6, 3.62, 11, 0.45, size=13,
    color=RGBColor(0xAA,0xCC,0xFF))

txt(s1, "이은정  |  rlojeong@catholic.ac.kr  |  가톨릭대학교",
    0.6, 5.75, 9, 0.38, size=12,
    color=RGBColor(0x70,0x90,0xB0))
txt(s1, datetime.date.today().strftime("%Y년 %m월 %d일"),
    0.6, 6.15, 6, 0.35, size=11,
    color=RGBColor(0x50,0x70,0x90))

# 오른쪽 stat 카드
card_data = [
    ("UE 수", "2 UEs"),
    ("KPI 피처", "6 features"),
    ("SNR Gap", "~4 dB"),
    ("Model", "Chronos TCN"),
]
for i, (k, v) in enumerate(card_data):
    lx, ty = 10.3, 1.9 + i*0.85
    rect(s1, lx, ty, 2.7, 0.72, fill=RGBColor(0x14,0x2A,0x50),
         line_color=C_LBLUE, line_w=15000)
    txt(s1, k, lx+0.15, ty+0.05, 2.4, 0.25, size=9,
        color=RGBColor(0x7B,0xB3,0xFF))
    txt(s1, v, lx+0.15, ty+0.3,  2.4, 0.35, size=14,
        bold=True, color=C_WHITE)


# ════════════════════════════════════════════════════════════════════
# SLIDE 2 — Project Overview
# ════════════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(BLANK)
rect(s2, 0, 0, 13.33, 7.5, fill=C_LGRAY)
header(s2, "Project Overview",
       "AI-based PRB Scheduling via Channel KPI Prediction in O-RAN")

# 목표 박스
rect(s2, 0.35, 1.28, 5.8, 2.9, fill=C_WHITE,
     line_color=C_BLUE, line_w=18000)
rect(s2, 0.35, 1.28, 5.8, 0.46, fill=C_BLUE)
txt(s2, "목표 (Objective)", 0.52, 1.32, 5.5, 0.38,
    size=13, bold=True, color=C_WHITE)
goals = [
    "OAI rfsimulator 기반 5G NR 테스트베드 구축",
    "UE별 채널 KPI (SNR/BLER/nPRB) 실시간 수집",
    "Chronos TCN 모델로 다음 스텝 KPI 예측",
    "공정성 가중치 → dApp → E3 → DU MAC 스케줄러",
    "TDL-C 채널 비대칭으로 실측 기반 평가",
]
for i, g in enumerate(goals):
    txt(s2, f"▸  {g}", 0.5, 1.84+i*0.46, 5.5, 0.42, size=11, color=C_DARK)

# 마일스톤 타임라인
rect(s2, 6.5, 1.28, 6.5, 2.9, fill=C_WHITE,
     line_color=C_GRAY, line_w=12000)
rect(s2, 6.5, 1.28, 6.5, 0.46, fill=C_GRAY)
txt(s2, "마일스톤", 6.68, 1.32, 6.2, 0.38,
    size=13, bold=True, color=C_WHITE)

milestones = [
    (C_GREEN,  "✅", "Namespace 격리 환경 구성"),
    (C_GREEN,  "✅", "TDL-C 채널 차별화 + OAI 소스 패치"),
    (C_GREEN,  "✅", "KPI 수집 파이프라인 (3,508행)"),
    (C_GREEN,  "✅", "Chronos TCN 학습 완료"),
    (C_GREEN,  "✅", "dApp 가중치 계산 실시간 동작"),
    (C_ORANGE, "🔧", "E3 → MAC 스케줄러 연동"),
]
for i, (col, icon, text) in enumerate(milestones):
    ty = 1.84 + i*0.39
    rect(s2, 6.62, ty+0.04, 0.22, 0.22, fill=col)
    txt(s2, text, 6.96, ty, 5.8, 0.38, size=11, color=C_DARK)

# 진행률 바
rect(s2, 0.35, 4.38, 12.65, 0.9, fill=C_WHITE,
     line_color=C_GRAY, line_w=10000)
txt(s2, "전체 진행률", 0.55, 4.46, 3, 0.3, size=11, bold=True, color=C_GRAY)
txt(s2, "83%", 11.8, 4.46, 1, 0.3, size=11, bold=True, color=C_BLUE)
rect(s2, 0.55, 4.82, 12.25, 0.32, fill=RGBColor(0xDD,0xE8,0xFF))
rect(s2, 0.55, 4.82, 10.15, 0.32, fill=C_BLUE)
txt(s2, "5/6 완료", 5.2, 4.84, 2, 0.28, size=10, bold=True,
    color=C_WHITE, align=PP_ALIGN.CENTER)

# 스택 카드 3개
stack = [
    ("OAI rfsimulator\n+ 5GC Docker", C_BLUE),
    ("Chronos TCN\n(BigDL)", C_LBLUE),
    ("dApp Controller\n(Python)", C_CYAN),
]
for i, (t, c) in enumerate(stack):
    lx = 0.35 + i*4.33
    rect(s2, lx, 5.45, 4.05, 1.78, fill=c)
    txt(s2, t, lx+0.2, 5.72, 3.6, 0.85,
        size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════
# SLIDE 3 — This Week's Changes
# ════════════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(BLANK)
rect(s3, 0, 0, 13.33, 7.5, fill=C_LGRAY)
header(s3, "This Week's Changes",
       "소스 패치 · 채널 설정 · 데이터 파이프라인 · dApp 구현")

# 좌측: 소스 수정
rect(s3, 0.35, 1.25, 6.1, 4.35, fill=C_WHITE,
     line_color=C_BLUE, line_w=16000)
rect(s3, 0.35, 1.25, 6.1, 0.42, fill=C_BLUE)
txt(s3, "OAI 소스 패치 (C/C++)", 0.52, 1.28, 5.8, 0.34,
    size=12, bold=True, color=C_WHITE)

patches = [
    ("random_channel.c", [
        "Doppler_phase_cur calloc 누락 → segfault 수정",
        "tdl_delays 원본 배열 수정 버그 → scaled_delays 분리",
        "get_normalization_ch_factor() randominit() 추가",
    ]),
    ("rangen_double.c", [
        "난수 초기화 순서 수정",
    ]),
    ("simulator.cpp (rfsimulator)", [
        "init_channelmod() 전 randominit() 추가",
        "→ chanmod 옵션 시 채널 초기화 정상화",
    ]),
]
ty = 1.77
for fname, lines in patches:
    txt(s3, f"📄 {fname}", 0.5, ty, 5.7, 0.3,
        size=10, bold=True, color=C_BLUE)
    ty += 0.3
    for line in lines:
        txt(s3, f"   • {line}", 0.5, ty, 5.7, 0.3, size=10, color=C_DARK)
        ty += 0.3
    ty += 0.08

# 우측: 신규 파일
rect(s3, 6.85, 1.25, 6.1, 4.35, fill=C_WHITE,
     line_color=C_LBLUE, line_w=16000)
rect(s3, 6.85, 1.25, 6.1, 0.42, fill=C_LBLUE)
txt(s3, "신규 파일 추가 (Python + Shell)", 7.02, 1.28, 5.8, 0.34,
    size=12, bold=True, color=C_WHITE)

new_files = [
    ("chronos_train.py",         "Chronos TCN 학습 (300 epoch)"),
    ("chronos_final.py",         "경량화 학습 버전"),
    ("chronos_live.py",          "live 데이터 재학습"),
    ("dapp_controller_chronos.py","TCN 기반 dApp 컨트롤러"),
    ("collect_kpi.py/live.py",   "gNB 로그 → CSV 파이프라인"),
    ("eval_dapp.py",             "TCN vs Persistence 비교"),
    ("fairness.py",              "Jain's Index 계산"),
    ("setup_namespaces.sh",      "ue1/ue2 네임스페이스 생성"),
    ("start_gnb/ue1/ue2.sh",     "실행 스크립트"),
    ("channelmod_rfsimu_ue_diff.conf","TDL-C 비대칭 채널 설정"),
]
for i, (f, desc) in enumerate(new_files):
    bg = C_LGRAY if i % 2 == 0 else C_WHITE
    rect(s3, 6.88, 1.75+i*0.383, 6.04, 0.37, fill=bg)
    txt(s3, f, 7.0, 1.78+i*0.383, 2.4, 0.3, size=9.5, bold=True, color=C_BLUE)
    txt(s3, desc, 9.45, 1.78+i*0.383, 3.3, 0.3, size=9.5, color=C_DARK)

# 하단 요약
rect(s3, 0.35, 5.75, 12.6, 1.48, fill=C_NAVY)
summary_stats = [
    ("수정 파일", "3개 C/C++"),
    ("신규 파일", "10개 Python/Shell"),
    ("수집 데이터", "3,508 + 749행"),
    ("학습 완료", "300 epochs"),
    ("val_loss", "0.000865"),
]
for i, (k, v) in enumerate(summary_stats):
    lx = 0.7 + i*2.48
    txt(s3, k, lx, 5.85, 2.2, 0.3, size=9,
        color=RGBColor(0x7B,0xB3,0xFF))
    txt(s3, v, lx, 6.18, 2.2, 0.45, size=18,
        bold=True, color=C_WHITE)


# ════════════════════════════════════════════════════════════════════
# SLIDE 4 — TX → SNR Performance
# ════════════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(BLANK)
rect(s4, 0, 0, 13.33, 7.5, fill=C_LGRAY)
header(s4, "TX-based SNR Differentiation",
       "TDL-C 채널 모델 비대칭 설정 → 실측 SNR/BLER 차이 확인")

# 채널 설정표
rect(s4, 0.35, 1.25, 6.0, 3.5, fill=C_WHITE,
     line_color=C_BLUE, line_w=16000)
rect(s4, 0.35, 1.25, 6.0, 0.42, fill=C_BLUE)
txt(s4, "channelmod_rfsimu_ue_diff.conf 설정", 0.52, 1.28, 5.7, 0.34,
    size=12, bold=True, color=C_WHITE)

params = [
    ("파라미터",       "UE0 (좋은 채널)",  "UE1 (나쁜 채널)"),
    ("type",          "TDL_C",            "TDL_C"),
    ("ploss_dB",      "0",                "0"),
    ("noise_power_dB","-40 dB",           "-20 dB   ← 20dB↑"),
    ("forgetfact",    "0.99",             "0.99"),
    ("ds_tdl",        "10 ns",            "10 ns"),
]
col_w = [2.2, 1.7, 1.7]
col_x = [0.4, 2.65, 4.38]
for r, row in enumerate(params):
    bg = C_BLUE if r==0 else (C_LGRAY if r%2==0 else C_WHITE)
    rect(s4, 0.38, 1.75+r*0.415, 5.92, 0.4, fill=bg)
    for c, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
        fc = C_WHITE if r==0 else C_DARK
        if r>0 and c==2: fc = C_ORANGE
        txt(s4, cell, cx+0.05, 1.78+r*0.415, cw, 0.34,
            size=10, bold=(r==0), color=fc,
            align=(PP_ALIGN.CENTER if c>0 else PP_ALIGN.LEFT))

# 실측 결과표
rect(s4, 6.7, 1.25, 6.25, 3.5, fill=C_WHITE,
     line_color=C_GREEN, line_w=16000)
rect(s4, 6.7, 1.25, 6.25, 0.42, fill=C_GREEN)
txt(s4, "실측 KPI (kpi_live.csv, 오늘)", 6.88, 1.28, 6.0, 0.34,
    size=12, bold=True, color=C_WHITE)

result_rows = [
    ("메트릭",     "RNTI 022d",    "RNTI 1096"),
    ("역할",       "UE0 (양호)",   "UE1 (불량)"),
    ("행 수",      "32행",         "738행"),
    ("평균 SNR",   "42.5 dB",      "38.4 dB"),
    ("SNR Gap",    "—",            "▼ 4.1 dB"),
    ("평균 BLER",  "0.0017",       "0.0960"),
    ("BLER 배율",  "1×",           "56× 나쁨"),
    ("평균 nPRB",  "106",          "106"),
]
col_xr = [6.75, 9.0, 10.8]
col_wr = [2.2, 1.75, 2.1]
for r, row in enumerate(result_rows):
    bg = C_GREEN if r==0 else (C_LGRAY if r%2==0 else C_WHITE)
    rect(s4, 6.73, 1.75+r*0.38, 6.18, 0.37, fill=bg)
    for c, (cell, cx, cw) in enumerate(zip(row, col_xr, col_wr)):
        fc = C_WHITE if r==0 else C_DARK
        if r>0 and c==2 and r in [4,6]: fc = C_RED
        txt(s4, cell, cx+0.05, 1.77+r*0.38, cw, 0.32,
            size=10, bold=(r==0 or r==1),
            color=fc,
            align=(PP_ALIGN.CENTER if c>0 else PP_ALIGN.LEFT))

# 하단: 막대 시각화
rect(s4, 0.35, 4.95, 12.6, 2.3, fill=C_WHITE,
     line_color=C_GRAY, line_w=10000)
txt(s4, "SNR & BLER 시각화", 0.55, 5.02, 5, 0.3,
    size=11, bold=True, color=C_NAVY)

# SNR 바
txt(s4, "SNR", 0.55, 5.42, 1, 0.28, size=10, bold=True, color=C_GRAY)
txt(s4, "022d  42.5 dB", 1.45, 5.42, 2.5, 0.28, size=10, color=C_DARK)
rect(s4, 1.45, 5.42, 4.25, 0.28, fill=RGBColor(0xDD,0xEE,0xFF))
rect(s4, 1.45, 5.42, 4.25, 0.28, fill=C_GREEN)

txt(s4, "1096  38.4 dB", 1.45, 5.78, 2.5, 0.28, size=10, color=C_DARK)
rect(s4, 1.45, 5.78, 4.25, 0.28, fill=RGBColor(0xDD,0xEE,0xFF))
rect(s4, 1.45, 5.78, 3.84, 0.28, fill=C_BLUE)

# BLER 바 (log scale 느낌으로 비율로)
txt(s4, "BLER", 6.7, 5.42, 1, 0.28, size=10, bold=True, color=C_GRAY)
txt(s4, "022d  0.0017", 7.6, 5.42, 2.5, 0.28, size=10, color=C_DARK)
rect(s4, 7.6, 5.42, 5.0, 0.28, fill=RGBColor(0xDD,0xEE,0xFF))
rect(s4, 7.6, 5.42, 0.09, 0.28, fill=C_GREEN)

txt(s4, "1096  0.0960", 7.6, 5.78, 2.5, 0.28, size=10, color=C_DARK)
rect(s4, 7.6, 5.78, 5.0, 0.28, fill=RGBColor(0xDD,0xEE,0xFF))
rect(s4, 7.6, 5.78, 5.0, 0.28, fill=C_RED)

txt(s4, "→ 채널 비대칭 확인: BLER 56배 차이, SNR 4.1 dB Gap",
    0.55, 6.3, 12, 0.35, size=11, bold=True, color=C_BLUE)


# ════════════════════════════════════════════════════════════════════
# SLIDE 5 — Troubleshooting
# ════════════════════════════════════════════════════════════════════
s5 = prs.slides.add_slide(BLANK)
rect(s5, 0, 0, 13.33, 7.5, fill=C_LGRAY)
header(s5, "Troubleshooting & Issues",
       "OAI 소스 버그 · 채널 초기화 · 평가 스크립트 오류")

issues = [
    {
        "status": "FIXED",
        "col": C_GREEN,
        "title": "TDL-C 채널 초기화 segfault",
        "cause": "random_channel.c: Doppler_phase_cur 포인터 미할당 상태로 접근",
        "fix":   "tdlModel() 내 calloc(nb_rx, sizeof(float)) 추가",
        "file":  "openair1/SIMULATION/TOOLS/random_channel.c",
    },
    {
        "status": "FIXED",
        "col": C_GREEN,
        "title": "tdl_delays 원본 배열 덮어쓰기 버그",
        "cause": "tdl_delays[i] *= DS_TDL → 원본 포인터 수정으로 재사용 시 오류",
        "fix":   "scaled_delays[] 별도 배열 할당 후 chan_desc->delays에 대입",
        "file":  "openair1/SIMULATION/TOOLS/random_channel.c",
    },
    {
        "status": "FIXED",
        "col": C_GREEN,
        "title": "randominit() 미호출로 채널 난수 고정",
        "cause": "chanmod 초기화 전 randominit() 미호출 → 모든 채널이 동일 시퀀스",
        "fix":   "simulator.cpp & random_channel.c 두 곳에 randominit() 추가",
        "file":  "radio/rfsimulator/simulator.cpp",
    },
    {
        "status": "FIXED",
        "col": C_GREEN,
        "title": "chronos_live.py MAE 출력 크래시",
        "cause": "float(e) — numpy array에 length-1 이상 원소 → TypeError",
        "fix":   "float(e) → float(np.mean(e)) 또는 인덱싱으로 수정 필요",
        "file":  "chronos_live.py  line 59",
    },
    {
        "status": "TODO",
        "col": C_ORANGE,
        "title": "E3 인터페이스 미구현",
        "cause": "OAI가 E3를 공식 지원 안 함 → dApp 가중치가 파일에만 기록",
        "fix":   "방안 1: OAI MAC 소스 패치 (파일 읽기)  /  방안 2: FlexRIC E2 RC SM",
        "file":  "dapp_controller_chronos.py → OAI MAC scheduler",
    },
    {
        "status": "TODO",
        "col": C_ORANGE,
        "title": "Global LSTM 스크립트 미완성",
        "cause": "train_rnn_global.py / dapp_controller_global.py 파일이 비어 있음",
        "fix":   "GlobalChannelRNN 모델 구현 및 Attention 레이어 추가",
        "file":  "train_rnn_global.py  (0 bytes)",
    },
]

for i, iss in enumerate(issues):
    row, col_idx = divmod(i, 2)
    lx = 0.35 + col_idx*6.5
    ty = 1.25 + row*2.0
    rect(s5, lx, ty, 6.15, 1.85, fill=C_WHITE,
         line_color=iss["col"], line_w=16000)
    rect(s5, lx, ty, 1.1, 0.36, fill=iss["col"])
    txt(s5, iss["status"], lx+0.06, ty+0.05, 1.0, 0.26,
        size=10, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    txt(s5, iss["title"], lx+1.18, ty+0.04, 4.8, 0.3,
        size=11, bold=True, color=C_DARK)
    txt(s5, f"원인: {iss['cause']}", lx+0.12, ty+0.44, 5.9, 0.38,
        size=9.5, color=C_GRAY)
    txt(s5, f"해결: {iss['fix']}", lx+0.12, ty+0.82, 5.9, 0.55,
        size=9.5, color=C_DARK)
    txt(s5, iss["file"], lx+0.12, ty+1.52, 5.9, 0.25,
        size=8.5, italic=True,
        color=RGBColor(0x60,0x80,0xA0))


# ════════════════════════════════════════════════════════════════════
# SLIDE 6 — Technical Architecture
# ════════════════════════════════════════════════════════════════════
s6 = prs.slides.add_slide(BLANK)
rect(s6, 0, 0, 13.33, 7.5, fill=C_LGRAY)
header(s6, "Technical Architecture",
       "O-RAN Testbed Stack · dApp 통합 위치 · 채널 예측 모듈")

# 왼쪽: O-RAN 스택 다이어그램
layers = [
    (C_GRAY,   "Non-RT RIC",          "rApp / 정책 관리"),
    (C_BLUE,   "Near-RT RIC",         "FlexRIC  (127.0.0.1:36422)  E2 Agent 연결"),
    (C_LBLUE,  "O-DU (OAI gNB)",      "Band 78 · 106 PRB · rfsim · E2 Agent enabled"),
    (C_CYAN,   "dApp Controller",      "← 현재 위치  |  E3 via /tmp/dapp_weights.json [임시]"),
    (C_GREEN,  "MAC Scheduler",        "PRB 할당  (dApp 가중치 미연동 ← TODO)"),
    (RGBColor(0x5,0x8A,0x8A), "UE1 / UE2 Namespace",
                               "oaitun_ue1/ue2  ·  iperf3 UDP 20Mbps"),
]
for i, (col, title, sub) in enumerate(layers):
    ty = 1.22 + i*1.02
    rect(s6, 0.35, ty, 5.6, 0.88, fill=col)
    txt(s6, title, 0.52, ty+0.06, 5.3, 0.32,
        size=12, bold=True, color=C_WHITE)
    txt(s6, sub,   0.52, ty+0.42, 5.3, 0.38,
        size=9.5, color=RGBColor(0xDD,0xEE,0xFF))
    if i < len(layers)-1:
        arrow = "E2 ↕" if i==1 else ("E3 ↕ [미구현]" if i==2 else "↕")
        acol = C_ORANGE if "미구현" in arrow else C_LBLUE
        txt(s6, arrow, 2.2, ty+0.88, 1.5, 0.15,
            size=8, bold=True, color=acol)

# 오른쪽: 채널 예측 모듈 흐름
rect(s6, 6.3, 1.22, 6.7, 6.0, fill=C_WHITE,
     line_color=C_GRAY, line_w=12000)
txt(s6, "채널 예측 모듈 상세 흐름", 6.5, 1.28, 6.4, 0.32,
    size=12, bold=True, color=C_NAVY)

flow_boxes = [
    (C_BLUE,   "OAI gNB 로그\n(gnb_live.log)",        "실시간 스트림"),
    (C_LBLUE,  "collect_kpi.py\n파싱 & CSV 저장",      "SNR/BLER/nPRB/MCS/bytes"),
    (C_CYAN,   "Chronos TCN\n(BigDL, 300 epoch)",      "lookback=10 → horizon=1"),
    (C_GREEN,  "공정성 가중치 계산\nscore = (1/SNR)×(1+BLER×5)",
                                                        "weight = score / Σscores"),
    (C_ORANGE, "/tmp/dapp_weights.json\n{022d:0.48, 1096:0.52}",
                                                        "1초 주기 갱신"),
]
for i, (col, title, sub) in enumerate(flow_boxes):
    ty = 1.68 + i*1.02
    rect(s6, 6.5, ty, 6.3, 0.82, fill=col)
    txt(s6, title, 6.65, ty+0.06, 5.5, 0.42,
        size=11, bold=True, color=C_WHITE)
    txt(s6, sub, 6.65, ty+0.5, 5.5, 0.28,
        size=9, color=RGBColor(0xEE,0xF6,0xFF))
    if i < len(flow_boxes)-1:
        txt(s6, "▼", 9.4, ty+0.82, 0.5, 0.2,
            size=12, bold=True, color=C_GRAY, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════
# SLIDE 7 — Next Steps & Timeline
# ════════════════════════════════════════════════════════════════════
s7 = prs.slides.add_slide(BLANK)
rect(s7, 0, 0, 13.33, 7.5, fill=C_LGRAY)
header(s7, "Next Steps & Timeline",
       "단기 / 중기 / 최종 마일스톤")

columns = [
    {
        "title": "단기 (1~2주)",
        "color": C_BLUE,
        "items": [
            "chronos_live.py 버그 수정\n(float(np.mean(e)))",
            "eval_dapp.py 실행 →\nMAE/RMSE 실측 수치 확보",
            "train_rnn_global.py 구현\n(GlobalChannelRNN LSTM+Attention)",
            "dapp_controller_global.py\n구현 완료",
            "Fairness 실험:\ndApp 적용 전/후 JFI 비교",
        ],
    },
    {
        "title": "중기 (3~4주)",
        "color": C_LBLUE,
        "items": [
            "OAI MAC 소스 패치:\n/tmp/dapp_weights.json 읽기",
            "실제 PRB 할당에 가중치\n반영 확인",
            "OpenVINO 추론 가속\n(fp32 최적화 검증)",
            "multi_att_collect.sh →\n다중 채널 환경 실험",
            "성능 비교:\nTCN vs LSTM vs Persistence",
        ],
    },
    {
        "title": "최종 마일스톤",
        "color": C_GREEN,
        "items": [
            "oai-channel-prediction/\nxApp 컨테이너화 (Docker)",
            "Kubernetes Helm Chart\n자동 배포",
            "E3 → DU 정식 연동\n(또는 FlexRIC E2 RC SM)",
            "논문 작성:\nAI-based Fairness in O-RAN",
            "공개 데이터셋 + 코드\nGitHub 릴리즈",
        ],
    },
]

for ci, col_data in enumerate(columns):
    lx = 0.35 + ci*4.33
    col = col_data["color"]
    rect(s7, lx, 1.22, 4.1, 0.5, fill=col)
    txt(s7, col_data["title"], lx+0.1, 1.27, 3.9, 0.38,
        size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    rect(s7, lx, 1.72, 4.1, 5.55, fill=C_WHITE,
         line_color=col, line_w=16000)
    for i, item in enumerate(col_data["items"]):
        ty = 1.82 + i*1.02
        rect(s7, lx+0.12, ty, 0.26, 0.26, fill=col)
        txt(s7, item, lx+0.5, ty-0.04, 3.45, 0.9, size=11, color=C_DARK)

# 하단: 핵심 성과
rect(s7, 0.35, 6.88, 12.6, 0.45, fill=C_NAVY)
txt(s7,
    "핵심 목표: 실측 KPI 기반 Chronos TCN 채널 예측 → 공정성 PRB 스케줄링 → O-RAN E3 연동",
    0.55, 6.92, 12.2, 0.34, size=11, bold=True, color=C_WHITE,
    align=PP_ALIGN.CENTER)


# ── 저장 ────────────────────────────────────────────────────────────
prs.save(OUT)
print(f"저장: {OUT}")
