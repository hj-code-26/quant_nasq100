"""수정 전/후 A·B 검증 — 같은 시나리오 파일을 두 커밋에서 돌려 비교한다.

    python research/sell_execution_audit/ab_verify.py [--base 6a0d843]

하는 일
  1. `git worktree` 로 base 커밋을 임시 디렉터리에 체크아웃한다 (작업 트리 무변경).
  2. 이 폴더(scenarios.py 포함)를 그 워크트리에 복사한다 — base 에는 없는 폴더다.
  3. 양쪽에서 `scenarios.py --json` 을 **별도 프로세스**로 돌린다.
  4. 결과를 표로 비교하고, 운영 DB·로그가 그대로인지 해시로 확인한다.

시나리오 자체는 `research.isolation.guard()` 아래에서 돈다 — 자격증명 제거,
운영 DB/로그 쓰기 차단, 토스 호스트 차단. 실주문·취소·정정은 가짜 브로커로만 간다.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WATCH = ["trading_decisions.db", "autotrade.log", ".toss_token.json", ".env",
         "autotrade.py", "toss.py"]


def fingerprint():
    out = {}
    for name in WATCH:
        p = ROOT / name
        if not p.exists():
            out[name] = None
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[name] = [p.stat().st_size, h[:16]]      # 비밀값은 출력하지 않는다 (해시 앞 16자)
    return out


def run(cwd):
    r = subprocess.run([sys.executable, "research/sell_execution_audit/scenarios.py", "--json"],
                       cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    line = next((x for x in reversed((r.stdout or "").splitlines()) if x.startswith("[")), None)
    if line is None:
        return None, (r.stdout or "")[-800:] + "\n--- stderr ---\n" + (r.stderr or "")[-1500:]
    return json.loads(line), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="6a0d843",
                    help="수정 직전 기준 커밋 (기본: 원래 분석 기준)")
    a = ap.parse_args()

    before = fingerprint()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ab-base-"))
    wt = tmp / "tree"
    subprocess.run(["git", "worktree", "add", "-f", "--detach", str(wt), a.base],
                   cwd=str(ROOT), check=True, capture_output=True)
    try:
        dst = wt / "research" / "sell_execution_audit"
        shutil.copytree(HERE, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        base_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(wt),
                                   capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
        base_res, base_err = run(wt)
        head_res, head_err = run(ROOT)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                       cwd=str(ROOT), capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"BASE = {base_head}   ({a.base})")
    print(f"HEAD = {head}\n")
    if base_err:
        print("base 실행 실패:\n" + base_err)
    if head_err:
        print("head 실행 실패:\n" + head_err)
    if not (base_res and head_res):
        return 2

    bmap = {r["id"]: r for r in base_res}
    fixed = regressed = same_ok = same_bad = 0
    print(f"{'ID':5} {'BASE':6} {'HEAD':6}  판정      시나리오")
    print("-" * 100)
    for h in head_res:
        b = bmap.get(h["id"])
        bo = b["ok"] if b else None
        verdict = ("FIXED   " if (bo is False and h["ok"]) else
                   "REGRESS " if (bo is True and not h["ok"]) else
                   "ok      " if h["ok"] else "STILL   ")
        fixed += verdict.startswith("FIXED")
        regressed += verdict.startswith("REGRESS")
        same_ok += verdict.startswith("ok")
        same_bad += verdict.startswith("STILL")
        print(f"{h['id']:5} {'PASS' if bo else 'FAIL':6} "
              f"{'PASS' if h['ok'] else 'FAIL':6}  {verdict}  {h['title']}")
        if verdict.startswith(("FIXED", "STILL", "REGRESS")):
            if b:
                print(f"{'':20}base: {b['detail'][:150]}")
            print(f"{'':20}head: {h['detail'][:150]}")

    print(f"\n수정으로 통과 전환 {fixed}건 · 원래도 통과 {same_ok}건 · "
          f"여전히 실패 {same_bad}건 · 회귀 {regressed}건")

    after = fingerprint()
    ch = [k for k in WATCH if before.get(k) != after.get(k)]
    print("[격리 확인] 운영 파일 " + ("변경 없음: " + ", ".join(WATCH) if not ch
                                 else "변경됨! " + ", ".join(ch)))
    return 1 if (regressed or ch) else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
