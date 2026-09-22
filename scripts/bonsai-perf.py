#!/usr/bin/env python3
"""Run sequential benchmark jobs from a JSON specification with bounded lifetimes."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return dt.datetime.now(dt.timezone.utc)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def terminate_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def power_state():
    if os.name != "nt":
        return None
    result = subprocess.run([
        "powershell.exe", "-NoProfile", "-Command",
        "Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | Select-Object PowerOnline,Discharging | ConvertTo-Json -Compress",
    ], capture_output=True, text=True, timeout=20, check=False)
    return {"returncode": result.returncode, "output": result.stdout.strip(), "error": result.stderr.strip()}


def run(spec_path):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat(spec["deadline_utc"])
    if deadline.utcoffset() is None:
        raise ValueError("deadline_utc must include a timezone")
    jobs = spec["jobs"]
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("job names must be unique file names")
    environment = os.environ.copy()
    environment.update(spec.get("env", {}))
    summary = {"started_utc": now().isoformat(), "spec": spec, "jobs": []}
    write_json(output / "summary.json", summary)
    for job in jobs:
        record_path = output / (job["name"] + ".json")
        if record_path.exists():
            raise FileExistsError(record_path)
        record = {"name": job["name"], "argv": job["argv"], "started_utc": now().isoformat(), "source_commit": job.get("source_commit")}
        record["power_before"] = power_state()
        timeout = min(float(job["timeout_seconds"]), (deadline - now()).total_seconds())
        if timeout <= 0:
            record["status"] = "deadline"
            record["returncode"] = None
        else:
            executable = Path(job["argv"][0])
            if executable.is_file():
                record["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            started = time.monotonic()
            with (output / (job["name"] + ".log")).open("wb") as log:
                process = subprocess.Popen(job["argv"], cwd=spec["cwd"], env=environment, stdout=log, stderr=subprocess.STDOUT, **options)
                record["pid"] = process.pid
                record["status"] = "running"
                write_json(record_path, record)
                print(json.dumps(record), flush=True)
                try:
                    record["returncode"] = process.wait(timeout=timeout)
                    record["status"] = "passed" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    terminate_tree(process)
                    record["returncode"] = process.returncode
                    record["status"] = "timeout"
                finally:
                    terminate_tree(process)
            record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = now().isoformat()
        record["power_after"] = power_state()
        write_json(record_path, record)
        summary["jobs"].append(record)
        summary["finished_utc"] = now().isoformat()
        write_json(output / "summary.json", summary)
        print(json.dumps(record), flush=True)
        if record["status"] != "passed":
            return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    raise SystemExit(run(parser.parse_args().spec))
