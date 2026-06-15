"""슬라이드용 컬러 로그 — 터미널 실행 후 스크린샷."""

RESET = "\033[0m"
GRAY  = "\033[38;2;60;60;60m"      # "t = NN s" 라벨
DIM   = "\033[38;2;140;140;140m"   # 보조 정보 (validate 등)
RED   = "\033[38;2;209;72;55m"     # 경보 / 문제 발견
AMBER = "\033[38;2;188;136;48m"    # 진행 중 / 복구 작업
BLUE  = "\033[38;2;54;119;194m"    # 상태 변경 / 로드
TEAL  = "\033[38;2;31;157;120m"    # 정상 / 성공

events = [
    ( 0, "twin loaded · 12,847 gaussians, monitoring · KPI = 0.94",       BLUE),
    ( 9, "monitoring · KPI = 0.91",              TEAL),
    (11, "KPI = 0.78  ↓, drift detected",                        RED),
    (13, "re-collect data · 384 fresh sample, re-fit gaussians · 3 iters",  AMBER),
    (18, "twin updated · 12,901 gaussians, validate · 96 UEs cross-checked",      BLUE),
]

print()
for t, msg, color in events:
    print(f"  {GRAY}t = {t:>2} s{RESET}   {color}{msg}{RESET}")
print()