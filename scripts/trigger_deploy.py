#!/usr/bin/env python3
"""
Trigger the 'Deploy to DigitalOcean' GitHub Actions workflow via GitHub REST API.
"""

import json
import os
import sys
import urllib.request

GITHUB_REPO = "Vnnie-Mun/zuntenium"
TOKEN = os.environ.get("GITHUB_TOKEN", "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")

def trigger_deploy(ref="main", skip_tests=False, monitoring=False):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/deploy.yml/dispatches"
    payload = {
        "ref": ref,
        "inputs": {
            "skip_tests": skip_tests,
            "monitoring": monitoring
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"token {TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "python"
        }
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status == 204:
            print("🚀 Successfully triggered 'Deploy to DigitalOcean' workflow on GitHub Actions!")
            print(f"Watch live at: https://github.com/{GITHUB_REPO}/actions/workflows/deploy.yml")
            return True
        else:
            print(f"Failed to trigger workflow: HTTP {resp.status}")
            return False

if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else "main"
    trigger_deploy(ref)
