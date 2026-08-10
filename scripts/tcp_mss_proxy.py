#!/usr/bin/env python3
"""Bridge standard I/O to TCP while advertising a bounded receive MSS."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from collections.abc import Sequence


def connect(host: str, port: int, mss: int) -> socket.socket:
    errors: list[OSError] = []
    for family, socktype, protocol, _, address in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        connection = socket.socket(family, socktype, protocol)
        try:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, mss)
            connection.settimeout(15.0)
            connection.connect(address)
            connection.settimeout(None)
            return connection
        except OSError as error:
            errors.append(error)
            connection.close()
    detail = errors[-1] if errors else "no addresses resolved"
    raise ConnectionError(f"could not connect to {host}:{port}: {detail}")


def upload(connection: socket.socket) -> None:
    try:
        while block := os.read(sys.stdin.fileno(), 64 * 1024):
            connection.sendall(block)
        connection.shutdown(socket.SHUT_WR)
    except (BrokenPipeError, ConnectionResetError, OSError):
        return


def bridge(connection: socket.socket) -> None:
    sender = threading.Thread(target=upload, args=(connection,), daemon=True)
    sender.start()
    try:
        while block := connection.recv(64 * 1024):
            os.write(sys.stdout.fileno(), block)
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--mss", type=int, default=1400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 256 <= args.mss <= 1460:
        raise SystemExit("--mss must be in [256, 1460]")
    with connect(args.host, args.port, args.mss) as connection:
        bridge(connection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
