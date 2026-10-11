#!/usr/bin/env python3
"""
Set GitHub Actions secrets for DigitalOcean deployment using PyNaCl and GitHub REST API.
Usage:
  python3 scripts/set_github_secrets.py <DO_HOST> <DO_USER> <PATH_TO_PRIVATE_KEY>
"""

import base64
import json
import os
import sys
import urllib.request
from nacl import encoding, public

GITHUB_REPO = "Vnnie-Mun/zuntenium"
TOKEN = os.environ.get("GITHUB_TOKEN", "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")

def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    public_key_obj = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key_obj)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")

def set_secret(secret_name: str, secret_value: str) -> bool:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/public-key",
        headers={"Authorization": f"token {TOKEN}", "User-Agent": "python"}
    )
    with urllib.request.urlopen(req) as resp:
        key_data = json.loads(resp.read().decode())
        key_id = key_data["key_id"]
        public_key = key_data["key"]

    encrypted_val = encrypt_secret(public_key, secret_value)

    put_data = json.dumps({"encrypted_value": encrypted_val, "key_id": key_id}).encode("utf-8")
    put_req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/secrets/{secret_name}",
        data=put_data,
        method="PUT",
        headers={
            "Authorization": f"token {TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "python"
        }
    )
    with urllib.request.urlopen(put_req) as resp:
        return resp.status in (201, 204)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 scripts/set_github_secrets.py <DO_HOST> <DO_USER> <PATH_TO_PRIVATE_KEY_OR_KEY_STRING>")
        sys.exit(1)

    host = sys.argv[1].strip()
    user = sys.argv[2].strip()
    key_arg = sys.argv[3].strip()

    if os.path.exists(key_arg):
        with open(key_arg, "r") as f:
            ssh_key = f.read().strip()
    else:
        ssh_key = key_arg

    print(f"Setting secret DO_HOST = {host}...")
    if set_secret("DO_HOST", host):
        print("  ✓ DO_HOST configured successfully.")

    print(f"Setting secret DO_USER = {user}...")
    if set_secret("DO_USER", user):
        print("  ✓ DO_USER configured successfully.")

    print("Setting secret DO_SSH_KEY...")
    if set_secret("DO_SSH_KEY", ssh_key):
        print("  ✓ DO_SSH_KEY configured successfully.")

    print("\nAll secrets configured in GitHub Actions!")

if __name__ == "__main__":
    main()
