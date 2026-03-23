import socket
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scapy.all import AsyncSniffer
from scapy.compat import raw
from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding, Packet

from src.config import MONITOR_LOG_PATH, PROXY_HOST, PROXY_PORT
from src.model.inference_mtc_byte import OnlineMTCByteClassifier
from src.utils.monitor_store import append_prediction, reset_prediction_log


classifier = OnlineMTCByteClassifier(
    model_path=Path("artifacts/cnn1d_mtc_byte.pth"),
    label_path=Path("artifacts/labels_mtc_byte.json"),
    seq_len=1500,
)


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


def reduce_tcp(packet: Packet, n_bytes: int = 20) -> Packet:
    if TCP in packet:
        tcp_header_length = packet[TCP].dataofs * 32 / 8
        if tcp_header_length > n_bytes:
            packet[TCP].dataofs = 5
            del packet[TCP].options
            del packet[TCP].chksum
            del packet[IP].chksum
            packet = packet.__class__(bytes(packet))
    return packet


def pad_udp(packet: Packet) -> Packet:
    if UDP in packet:
        layer_after = packet[UDP].payload.copy()
        pad = Padding()
        pad.load = "\x00" * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

    return packet


def packet_to_array(packet: Packet, max_length: int = 1500) -> np.ndarray:
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    if len(arr) < max_length:
        arr = np.pad(arr, pad_width=(0, max_length - len(arr)), constant_values=0)
    return arr.astype(np.float32)


def filter_packet(pkt: Packet) -> Packet | None:
    if Ether in pkt:
        pkt = pkt[Ether].payload

    if IP in pkt:
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"
    else:
        return None

    if TCP in pkt:
        pkt = reduce_tcp(pkt)
    elif UDP in pkt:
        pkt = pad_udp(pkt)
    else:
        return None

    fin = 0x01
    syn = 0x02
    ack = 0x10

    ip_packet = pkt[IP]
    if TCP in ip_packet:
        tcp_packet = ip_packet[TCP]
        if tcp_packet.flags & 0x16 in [ack, syn, fin]:
            return None
    elif UDP in ip_packet:
        udp_packet = ip_packet[UDP]
        if udp_packet.dport == 53 or udp_packet.sport == 53 or DNS in pkt:
            return None
    else:
        return None

    return pkt


class PacketInferenceTap:
    def __init__(
        self,
        flow_id: str,
        local_ip: str,
        local_port: int,
        remote_ip: str,
        remote_port: int,
    ) -> None:
        self.flow_id = flow_id
        self.local_ip = local_ip
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.predicted = False
        self.lock = threading.Lock()
        self.sniffer: Optional[AsyncSniffer] = None

    def _matches_flow(self, pkt: Packet) -> bool:
        if IP not in pkt or TCP not in pkt:
            return False
        ip = pkt[IP]
        tcp = pkt[TCP]

        fwd = (
            str(ip.src) == self.local_ip
            and str(ip.dst) == self.remote_ip
            and int(tcp.sport) == self.local_port
            and int(tcp.dport) == self.remote_port
        )
        rev = (
            str(ip.src) == self.remote_ip
            and str(ip.dst) == self.local_ip
            and int(tcp.sport) == self.remote_port
            and int(tcp.dport) == self.local_port
        )
        return fwd or rev

    def _on_packet(self, pkt: Packet) -> None:
        if self.predicted:
            return
        if not self._matches_flow(pkt):
            return

        cleaned = filter_packet(pkt)
        if cleaned is None:
            return

        vec = packet_to_array(cleaned, max_length=1500)
        label = classifier.predict(vec)

        with self.lock:
            if self.predicted:
                return
            self.predicted = True
            print(f"[{self.flow_id}] Stream Type (MTC-Byte/CleanedPacket): {label}")
            append_prediction(MONITOR_LOG_PATH, self.flow_id, label)

    def start(self) -> None:
        bpf = (
            "tcp and "
            "((src host {lip} and dst host {rip} and src port {lp} and dst port {rp}) "
            "or (src host {rip} and dst host {lip} and src port {rp} and dst port {lp}))"
        ).format(lip=self.local_ip, rip=self.remote_ip, lp=self.local_port, rp=self.remote_port)

        try:
            self.sniffer = AsyncSniffer(filter=bpf, prn=self._on_packet, store=False)
            self.sniffer.start()
        except Exception as e:
            print(f"[{self.flow_id}] [WARN] Packet sniffer start failed: {e}")
            self.sniffer = None

    def stop(self) -> None:
        if self.sniffer is None:
            return
        try:
            self.sniffer.stop()
        except Exception:
            pass


def relay_bidirectional(client: socket.socket, remote: socket.socket, flow_id: str) -> None:
    client.settimeout(None)
    remote.settimeout(None)

    try:
        local_ip, local_port = remote.getsockname()
        remote_ip, remote_port = remote.getpeername()
    except OSError:
        local_ip, local_port = "", 0
        remote_ip, remote_port = "", 0

    tap = PacketInferenceTap(
        flow_id=flow_id,
        local_ip=str(local_ip),
        local_port=int(local_port),
        remote_ip=str(remote_ip),
        remote_port=int(remote_port),
    )
    tap.start()

    def forward(src: socket.socket, dst: socket.socket) -> None:
        while True:
            try:
                data = src.recv(4096)
            except (socket.timeout, TimeoutError, OSError):
                break
            if not data:
                break

            try:
                dst.sendall(data)
            except OSError:
                break

        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    t1 = threading.Thread(target=forward, args=(client, remote), daemon=True)
    t2 = threading.Thread(target=forward, args=(remote, client), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    tap.stop()


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
    reset_prediction_log(MONITOR_LOG_PATH)
    print(f"Prediction log reset: {MONITOR_LOG_PATH}")
    if not classifier.enabled:
        print("[WARN] MTC-byte model not ready. Check artifacts/cnn1d_mtc_byte.pth and labels_mtc_byte.json")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((PROXY_HOST, PROXY_PORT))
    server.listen(200)
    print(f"MTC-byte proxy listening on {PROXY_HOST}:{PROXY_PORT}")

    while True:
        client, addr = server.accept()
        threading.Thread(target=handle_client, args=(client, addr), daemon=True).start()


if __name__ == "__main__":
    run_proxy()
