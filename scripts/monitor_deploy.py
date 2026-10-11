#!/usr/bin/env python3
"""
Monitor GitHub Actions workflow run for Deploy to DigitalOcean with retry handling.
"""

import json
import os
import sys
import time
import urllib.request

GITHUB_REPO = "Vnnie-Mun/zuntenium"
TOKEN = os.environ.get("GITHUB_TOKEN", "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")

def api_get(url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"token {TOKEN}", "User-Agent": "python"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))

def get_run(run_id):
    return api_get(f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}")

def get_jobs(run_id):
    data = api_get(f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}/jobs")
    return data.get("jobs", [])

def monitor(run_id, timeout_sec=1200):
    start = time.time()
    print(f"Tracking run {run_id} on {GITHUB_REPO}...")

    last_status = None
    while time.time() - start < timeout_sec:
        try:
            data = get_run(run_id)
            status = data.get("status")
            conclusion = data.get("conclusion")

            jobs = get_jobs(run_id)
            job_summary = " | ".join([f"{j['name']}: {j['status']}({j.get('conclusion') or '-'})" for j in jobs])

            current_state = f"Status: {status} | Conclusion: {conclusion} | [{job_summary}]"
            if current_state != last_status:
                print(f"[{int(time.time() - start)}s] {current_state}")
                last_status = current_state

            if status == "completed":
                print(f"\nWorkflow finished with conclusion: {conclusion}")
                for j in jobs:
                    print(f"\n--- Job: {j['name']} ({j['conclusion']}) ---")
                    for s in j.get("steps", []):
                        icon = "✓" if s.get("conclusion") == "success" else ("✗" if s.get("conclusion") == "failure" else "○")
                        print(f"  {icon} {s['name']}: {s.get('conclusion') or s.get('status')}")
                return conclusion == "success"

        except Exception as e:
            print(f"[Warning] Transient error checking status: {e}")

        time.sleep(10)

    print("\nTimeout waiting for workflow run to finish.")
    return False

if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "38143193074"
    success = monitor(run_id)
    sys.exit(0 if success else 1)
