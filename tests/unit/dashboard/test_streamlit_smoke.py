from __future__ import annotations

import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "app" / "streamlit_app.py"
STARTUP_TIMEOUT_SECONDS = 15.0
POLL_INTERVAL_SECONDS = 0.1


def test_streamlit_server_starts_offline_and_reports_healthy() -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    port = _available_loopback_port()

    _assert_streamlit_server_starts(
        _streamlit_command(port),
        port=port,
        deadline=deadline,
    )


def test_streamlit_startup_rejects_health_from_another_listener() -> None:
    logs_before = _smoke_log_paths()

    with _loopback_server(_HealthyHandler) as port:
        start = time.monotonic()
        deadline = start + 3.0
        with pytest.raises(
            AssertionError,
            match=r"exited before becoming ready|did not report healthy",
        ):
            _assert_streamlit_server_starts(
                _streamlit_command(port),
                port=port,
                deadline=deadline,
            )

        assert time.monotonic() <= deadline

    assert _smoke_log_paths() == logs_before


def test_streamlit_startup_obeys_total_deadline_for_slow_response() -> None:
    logs_before = _smoke_log_paths()

    with _loopback_server(_SlowUnavailableHandler) as port:
        start = time.monotonic()
        deadline = start + 0.2
        with pytest.raises(
            AssertionError,
            match=r"exited before becoming ready|did not report healthy",
        ):
            _assert_streamlit_server_starts(
                [sys.executable, "-c", "import time; time.sleep(0.5)"],
                port=port,
                deadline=deadline,
            )

        assert time.monotonic() - start < 1.0

    assert _smoke_log_paths() == logs_before


def _streamlit_command(port: int) -> list[str]:
    return [
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


def _smoke_log_paths() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("urbanflow-streamlit-smoke-*.log"))


class _LoopbackServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_EXCLUSIVEADDRUSE,
                1,
            )
        super().server_bind()


@contextmanager
def _loopback_server(
    handler: type[BaseHTTPRequestHandler],
) -> Iterator[int]:
    server = _LoopbackServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
    )
    thread.start()
    try:
        yield int(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


class _HealthyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/_stcore/health":
            self.send_error(404)
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _SlowUnavailableHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"x" * 20
        self.send_response(503)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            for byte in body:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        return int(port_socket.getsockname()[1])


def _health_status(port: int, *, deadline: float) -> int | None:
    request = b"GET /_stcore/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as health_socket:
        health_socket.setblocking(False)
        connect_error = health_socket.connect_ex(("127.0.0.1", port))
        if connect_error:
            if not _wait_for_socket(
                health_socket,
                writable=True,
                deadline=deadline,
            ):
                return None
            if health_socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                return None

        sent = 0
        while sent < len(request):
            if not _wait_for_socket(
                health_socket,
                writable=True,
                deadline=deadline,
            ):
                return None
            try:
                sent += health_socket.send(request[sent:])
            except BlockingIOError:
                continue

        status_line = bytearray()
        while not status_line.endswith(b"\r\n"):
            if len(status_line) > 4096:
                return None
            if not _wait_for_socket(
                health_socket,
                readable=True,
                deadline=deadline,
            ):
                return None
            try:
                chunk = health_socket.recv(1)
            except BlockingIOError:
                continue
            if not chunk:
                return None
            status_line.extend(chunk)

    if time.monotonic() > deadline:
        return None
    try:
        protocol, status, _reason = status_line.decode("ascii").split(" ", 2)
        if not protocol.startswith("HTTP/"):
            return None
        return int(status)
    except (UnicodeDecodeError, ValueError):
        return None


def _wait_for_socket(
    target: socket.socket,
    *,
    deadline: float,
    readable: bool = False,
    writable: bool = False,
) -> bool:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    ready_to_read, ready_to_write, exceptional = select.select(
        [target] if readable else [],
        [target] if writable else [],
        [target],
        remaining,
    )
    return not exceptional and bool(ready_to_read or ready_to_write)


def _log_reports_selected_address(log_path: Path, *, port: int) -> bool:
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    expected_url = f"URL: http://127.0.0.1:{port}"
    return any(line.strip() == expected_url for line in log_text.splitlines())


def _delete_log(log_path: Path) -> None:
    delete_deadline = time.monotonic() + 0.5
    while True:
        try:
            log_path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= delete_deadline:
                raise
            time.sleep(0.01)


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

            attempt_deadline = min(deadline, time.monotonic() + 0.25)
            try:
                status = _health_status(port, deadline=attempt_deadline)
            except OSError:
                status = None
            if (
                status == 200
                and time.monotonic() <= deadline
                and _log_reports_selected_address(log_path, port=port)
            ):
                break

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
                        _delete_log(log_path)

    if log_path is None:
        raise AssertionError(failure or "Streamlit smoke logfile was not created.")

    if failure is not None:
        raise AssertionError(f"{failure}\n\nStreamlit log:\n{log_text}")
