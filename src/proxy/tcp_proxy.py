import socket
import threading
import time
from typing import Tuple

from src.config import MONITOR_LOG_PATH, PROXY_HOST, PROXY_PORT, SEQ_LEN
from src.features.packet_sequence import PacketSequenceBuffer
from src.model.inference import OnlineTrafficClassifier
from src.utils.monitor_store import append_prediction


classifier = OnlineTrafficClassifier()


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


def relay_bidirectional(client: socket.socket, remote: socket.socket, flow_id: str) -> None:
    feature_buf = PacketSequenceBuffer(seq_len=SEQ_LEN)
    predicted = False

    def forward(src: socket.socket, dst: socket.socket) -> None:
        nonlocal predicted
        while True:
            data = src.recv(4096)
            if not data:
                break

            feature_buf.add_packet_len(len(data))
            if feature_buf.is_ready() and not predicted:
                label = classifier.predict(feature_buf.to_normalized_vector())
                print(f"[{flow_id}] Stream Type: {label}")
                append_prediction(MONITOR_LOG_PATH, flow_id, label)
                predicted = True

            dst.sendall(data)

    t1 = threading.Thread(target=forward, args=(client, remote), daemon=True)
    t2 = threading.Thread(target=forward, args=(remote, client), daemon=True)
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


def run_proxy() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((PROXY_HOST, PROXY_PORT))
    server.listen(200)
    print(f"Proxy listening on {PROXY_HOST}:{PROXY_PORT}")

    while True:
        client, addr = server.accept()
        threading.Thread(target=handle_client, args=(client, addr), daemon=True).start()


if __name__ == "__main__":
    run_proxy()
