from __future__ import annotations

import http.client
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "app" / "streamlit_app.py"
STARTUP_TIMEOUT_SECONDS = 15.0
POLL_INTERVAL_SECONDS = 0.1


def test_streamlit_server_starts_offline_and_reports_healthy() -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    port = _available_loopback_port()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ENTRYPOINT.relative_to(ROOT)),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]

    _assert_streamlit_server_starts(command, port=port, deadline=deadline)


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        return int(port_socket.getsockname()[1])


def _assert_streamlit_server_starts(
    command: list[str],
    *,
    port: int,
    deadline: float,
) -> None:
    process: subprocess.Popen[bytes] | None = None
    failure: str | None = None
    log_path: Path | None = None
    log_file = None
    log_text = ""

    try:
        with tempfile.NamedTemporaryFile(
            prefix="urbanflow-streamlit-smoke-",
            suffix=".log",
            delete=False,
        ) as temporary_log:
            log_path = Path(temporary_log.name)
        log_file = log_path.open("wb")

        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            failure = f"Streamlit process could not start: {exc}"

        while process is not None and failure is None:
            return_code = process.poll()
            if return_code is not None:
                failure = f"Streamlit exited before becoming ready (exit {return_code})."
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = (
                    f"Streamlit did not report healthy within {STARTUP_TIMEOUT_SECONDS:g} seconds."
                )
                break

            connection = http.client.HTTPConnection(
                "127.0.0.1",
                port,
                timeout=min(0.25, remaining),
            )
            try:
                connection.request("GET", "/_stcore/health")
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    break
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()

            sleep_seconds = min(
                POLL_INTERVAL_SECONDS,
                max(0.0, deadline - time.monotonic()),
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)
    finally:
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            try:
                if log_file is not None:
                    log_file.close()
            finally:
                if log_path is not None:
                    try:
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                    finally:
                        log_path.unlink(missing_ok=True)

    if log_path is None:
        raise AssertionError(failure or "Streamlit smoke logfile was not created.")

    if failure is not None:
        raise AssertionError(f"{failure}\n\nStreamlit log:\n{log_text}")
