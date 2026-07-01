import subprocess
import sys
import os

# =========================
# 실험 목록 — 여기만 수정하면 됨
# =========================
runs = [
        {
        "model":       "Model_41",
        "tag":         "LOSO_30ch_st-gcn_run41_right_zaligned_fix",
        "subjects":    "right",
        "loader":      "zaligned_fix",   # train loader
        "test_loader": "zaligned_fix",         # test loader (augmentation 없이 고정)
    },
    {
        "model":       "Model_116",
        "tag":         "LOSO_30ch_st-gcn_run116_right_compliment",
        "subjects":    "right",
        "loader":      "simple",   # train loader
        "test_loader": "simple",         # test loader (augmentation 없이 고정)
    },
]

# =========================
# 순서대로 실행
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))

for i, cfg in enumerate(runs):
    print(f"\n{'='*60}")
    print(f"  RUN {i+1}/{len(runs)}: {cfg['tag']}")
    print(f"  model={cfg['model']}  subjects={cfg['subjects']}  loader={cfg['loader']}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "IMU_main_LOSO.py",
        "--model",       cfg["model"],
        "--tag",         cfg["tag"],
        "--subjects",    cfg["subjects"],
        "--loader",      cfg["loader"],
        "--test_loader", cfg.get("test_loader", "simple"),
    ]

    result = subprocess.run(cmd, cwd=base_dir)

    if result.returncode != 0:
        print(f"\n[ERROR] {cfg['tag']} 실패 (return code {result.returncode}). 중단.")
        sys.exit(result.returncode)

print(f"\n{'='*60}")
print(f"  모든 실험 완료! ({len(runs)}개)")
print(f"{'='*60}")
