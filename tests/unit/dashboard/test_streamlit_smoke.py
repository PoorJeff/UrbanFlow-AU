from __future__ import annotations

import select
import socket
import struct
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
CLEANUP_SLACK_SECONDS = 1.0
MAX_HEALTH_HEADER_BYTES = 16 * 1024


def test_streamlit_server_starts_offline_and_reports_healthy() -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    port = _available_loopback_port()

    _assert_streamlit_server_starts(
        _streamlit_command(port),
        port=port,
        deadline=deadline,
        timeout_seconds=STARTUP_TIMEOUT_SECONDS,
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
                timeout_seconds=3.0,
            )

        assert time.monotonic() - start <= 3.0 + CLEANUP_SLACK_SECONDS

    assert _smoke_log_paths() == logs_before


def test_health_status_rejects_200_with_incomplete_slow_headers() -> None:
    with _loopback_server(_SlowIncompleteHeadersHandler) as port:
        start = time.monotonic()
        deadline = start + 0.2

        assert _health_status(port, deadline=deadline) is None
        assert time.monotonic() <= deadline + 0.1


def test_health_status_returns_none_when_peer_resets_connection() -> None:
    if not hasattr(socket, "SO_LINGER"):
        pytest.skip("SO_LINGER is required to force an abortive close")

    with _abortive_loopback_server() as port:
        start = time.monotonic()
        deadline = start + 0.5

        assert _health_status(port, deadline=deadline) is None
        assert time.monotonic() <= deadline + 0.1


def test_health_status_does_not_read_slow_response_body() -> None:
    with _loopback_server(_SlowUnavailableBodyHandler) as port:
        deadline = time.monotonic() + 0.2

        assert _health_status(port, deadline=deadline) == 503
        assert time.monotonic() <= deadline


def test_streamlit_startup_obeys_deadline_with_bounded_cleanup() -> None:
    logs_before = _smoke_log_paths()
    polling_budget = 0.2

    port = _available_loopback_port()
    start = time.monotonic()
    deadline = start + polling_budget
    with pytest.raises(
        AssertionError,
        match=r"did not report healthy within 0\.2 seconds",
    ):
        _assert_streamlit_server_starts(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            port=port,
            deadline=deadline,
            timeout_seconds=polling_budget,
        )

    assert time.monotonic() - start <= polling_budget + CLEANUP_SLACK_SECONDS

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


@contextmanager
def _abortive_loopback_server() -> Iterator[int]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(1.0)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    cleanup_started = threading.Event()
    server_errors: list[BaseException] = []

    def reset_after_request() -> None:
        try:
            connection, _address = listener.accept()
            with connection:
                connection.settimeout(1.0)
                request = bytearray()
                while not request.endswith(b"\r\n\r\n"):
                    chunk = connection.recv(1024)
                    if not chunk:
                        raise AssertionError("health client closed before sending its request")
                    request.extend(chunk)
                linger_format = "HH" if sys.platform == "win32" else "ii"
                connection.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_LINGER,
                    struct.pack(linger_format, 1, 0),
                )
        except OSError as exc:
            if not cleanup_started.is_set():
                server_errors.append(exc)
        except BaseException as exc:
            server_errors.append(exc)

    thread = threading.Thread(target=reset_after_request)
    thread.start()
    try:
        yield int(listener.getsockname()[1])
    finally:
        cleanup_started.set()
        listener.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not server_errors


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


class _SlowIncompleteHeadersHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.wfile.write(b"HTTP/1.1 200 OK\r\n")
        self.wfile.flush()
        time.sleep(0.5)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _SlowUnavailableBodyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(503)
        self.send_header("Content-Length", "1")
        self.end_headers()
        self.wfile.flush()
        time.sleep(0.5)
        try:
            self.wfile.write(b"x")
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

    try:
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

            if time.monotonic() > deadline:
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

            response_headers = bytearray()
            while not response_headers.endswith(b"\r\n\r\n"):
                if len(response_headers) >= MAX_HEALTH_HEADER_BYTES:
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
                response_headers.extend(chunk)
    except OSError:
        return None

    if time.monotonic() > deadline:
        return None
    try:
        status_line = response_headers.partition(b"\r\n")[0]
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
    timeout_seconds: float,
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
                failure = f"Streamlit did not report healthy within {timeout_seconds:g} seconds."
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
