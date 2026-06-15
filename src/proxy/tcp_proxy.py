import argparse
import socket
import threading
import time
from pathlib import Path
from typing import Tuple

from src.config import (
    MONITOR_LOG_PATH,
    PROXY_HOST,
    PROXY_PORT,
    SEQ_LEN,
    SIGNED_LENGTHS,
    get_label_path,
    get_model_path,
    get_monitor_log_path,
)
from src.features.packet_sequence import PacketSequenceBuffer
from src.model.inference import OnlineTrafficClassifier
from src.utils.monitor_store import append_prediction, reset_prediction_log


def parse_target_from_connect(first_line: str) -> Tuple[str, int]:
    # CONNECT example: CONNECT example.com:443 HTTP/1.1
    _, host_port, _ = first_line.split(" ", 2)
    host, port = host_port.split(":")
    return host, int(port)


def parse_target_from_http(headers: str) -> Tuple[str, int]:
    host = ""
    for line in headers.split("\r\n"):
        if line.lower().startswith("host:"):
            host = line.split(":", 1)[1].strip()
            break
    if ":" in host:
        h, p = host.rsplit(":", 1)
        return h, int(p)
    return host, 80


def run_proxy(
    model_path: Path | None = None,
    label_path: Path | None = None,
    monitor_log_path: Path | None = None,
) -> None:
    _model_path = model_path or get_model_path()
    _label_path = label_path or get_label_path()
    _monitor_log = monitor_log_path or get_monitor_log_path()

    classifier = OnlineTrafficClassifier(
        model_path=_model_path,
        label_path=_label_path,
    )

    reset_prediction_log(_monitor_log)
    print(f"Prediction log reset: {_monitor_log}")

    def relay_bidirectional(client: socket.socket, remote: socket.socket, flow_id: str) -> None:
        feature_buf = PacketSequenceBuffer(seq_len=SEQ_LEN, signed_lengths=SIGNED_LENGTHS)
        predicted = False

        # Keep long-lived tunnels (e.g. HTTPS CONNECT) from failing on idle read timeout.
        client.settimeout(None)
        remote.settimeout(None)

        def forward(src: socket.socket, dst: socket.socket, direction: int) -> None:
            nonlocal predicted
            while True:
                try:
                    data = src.recv(4096)
                except (socket.timeout, TimeoutError, OSError):
                    break
                if not data:
                    break

                feature_buf.add_packet_len(len(data), direction=direction)
                if feature_buf.is_ready() and not predicted:
                    label = classifier.predict(feature_buf.to_normalized_vector())
                    print(f"[{flow_id}] Stream Type: {label}")
                    append_prediction(_monitor_log, flow_id, label)
                    predicted = True

                try:
                    dst.sendall(data)
                except OSError:
                    break

            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        t1 = threading.Thread(target=forward, args=(client, remote, 1), daemon=True)
        t2 = threading.Thread(target=forward, args=(remote, client, -1), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def handle_client(client: socket.socket, client_addr) -> None:
        flow_id = f"{client_addr[0]}:{client_addr[1]}-{int(time.time())}"
        remote = None
        try:
            req = client.recv(8192)
            if not req:
                return

            text = req.decode("latin1", errors="ignore")
            first_line = text.split("\r\n", 1)[0]

            if first_line.startswith("CONNECT "):
                host, port = parse_target_from_connect(first_line)
                remote = socket.create_connection((host, port), timeout=10)
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                relay_bidirectional(client, remote, flow_id)
            else:
                host, port = parse_target_from_http(text)
                remote = socket.create_connection((host, port), timeout=10)
                remote.sendall(req)
                relay_bidirectional(client, remote, flow_id)
        except Exception as e:
            print(f"[{flow_id}] Error: {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass
            if remote is not None:
                try:
                    remote.close()
                except Exception:
                    pass

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((PROXY_HOST, PROXY_PORT))
    server.listen(200)
    print(f"Proxy listening on {PROXY_HOST}:{PROXY_PORT}")

    while True:
        client, addr = server.accept()
        threading.Thread(target=handle_client, args=(client, addr), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="TCP proxy with traffic classification")
    parser.add_argument(
        "--tag",
        default=None,
        type=str,
        help="Experiment tag. Loads model from artifacts/<tag>/ and writes predictions there.",
    )
    args = parser.parse_args()

    run_proxy(
        model_path=get_model_path(args.tag),
        label_path=get_label_path(args.tag),
        monitor_log_path=get_monitor_log_path(args.tag),
    )


if __name__ == "__main__":
    main()
