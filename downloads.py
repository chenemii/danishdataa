#!/usr/bin/env python3
import os
import sys
import getpass
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import paramiko


HOST = os.environ.get("SFTP_HOST")
PORT = int(os.environ.get("SFTP_PORT", "22"))
USERNAME = os.environ.get("SFTP_USERNAME")        # Works even with '@' in username
REMOTE_DIR = os.environ.get("SFTP_REMOTE_DIR")
LOCAL_DIR = os.environ.get("SFTP_LOCAL_DIR", "./downloads")
MAX_WORKERS = 10                           # parallel downloads
FILTER_EXT = None                          


def ensure_local_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def list_remote_files(client: paramiko.SSHClient, remote_dir: str) -> List[str]:
    sftp = client.open_sftp()
    try:
        sftp.chdir(remote_dir)
        files = sftp.listdir()
        # keep only regular-looking files
        files = [f for f in files if not f.startswith(".")]
        files.sort()
        return files
    finally:
        sftp.close()


def sizes_match(sftp: paramiko.SFTPClient, remote_path: str, local_path: str) -> bool:
    try:
        rstat = sftp.stat(remote_path)
    except FileNotFoundError:
        return False
    if not os.path.exists(local_path):
        return False
    try:
        lsize = os.path.getsize(local_path)
    except OSError:
        return False
    return lsize == rstat.st_size


def download_one(host: str, port: int, username: str, password: str,
                 remote_dir: str, filename: str, local_dir: str) -> str:
    """
    Use a dedicated SFTP connection per thread (paramiko SFTPClient is not thread-safe).
    """
    transport = None
    sftp = None
    remote_path = f"{remote_dir}/{filename}"
    local_path = os.path.join(local_dir, filename)

    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Skip if already complete
        if sizes_match(sftp, remote_path, local_path):
            return f"SKIP (exists): {filename}"

        # If a partial file exists, remove and redownload (simple + robust)
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass

        sftp.get(remote_path, local_path)
        return f"OK: {filename}"
    except (paramiko.SSHException, socket.error) as e:
        return f"ERROR: {filename} -> {e}"
    finally:
        try:
            if sftp:
                sftp.close()
        finally:
            if transport:
                transport.close()


def main():
    # Validate required configuration to avoid leaking private defaults
    missing = []
    if not HOST:
        missing.append("SFTP_HOST")
    if not USERNAME:
        missing.append("SFTP_USERNAME")
    if not REMOTE_DIR:
        missing.append("SFTP_REMOTE_DIR")
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Set them and retry. Example:")
        print("  export SFTP_HOST='example.com'")
        print("  export SFTP_USERNAME='user@example.com'")
        print("  export SFTP_REMOTE_DIR='path/on/server'")
        sys.exit(1)

    print(f"Connecting to {HOST} as {USERNAME} ...")
    password = os.environ.get("SFTP_PASSWORD")
    if not password:
        # secure prompt
        password = getpass.getpass("SFTP password: ")

    # Base control connection (used only to list files)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # trust on first use; for production pin host key
    try:
        client.connect(HOST, port=PORT, username=USERNAME, password=password, look_for_keys=False, allow_agent=False)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    try:
        all_files = list_remote_files(client, REMOTE_DIR)
    finally:
        client.close()

    if FILTER_EXT:
        files = [f for f in all_files if f.endswith(FILTER_EXT)]
    else:
        files = all_files

    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files in {REMOTE_DIR}. Downloading to {LOCAL_DIR}")
    ensure_local_dir(LOCAL_DIR)

    # Download in parallel, up to MAX_WORKERS at a time
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                download_one, HOST, PORT, USERNAME, password, REMOTE_DIR, fname, LOCAL_DIR
            ): fname for fname in files
        }
        for fut in as_completed(futures):
            msg = fut.result()
            print(msg)
            results.append(msg)

    # Summary
    ok = sum(1 for r in results if r.startswith("OK:"))
    skipped = sum(1 for r in results if r.startswith("SKIP"))
    errors = [r for r in results if r.startswith("ERROR")]
    print(f"\nDone. OK={ok}, SKIP={skipped}, ERROR={len(errors)}")
    if errors:
        print("Some errors occurred:")
        for e in errors:
            print("  -", e)


if __name__ == "__main__":
    main()

