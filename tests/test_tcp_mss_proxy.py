from __future__ import annotations

import importlib.util
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/tcp_mss_proxy.py"
SPEC = importlib.util.spec_from_file_location("tcp_mss_proxy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tcp_mss_proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tcp_mss_proxy)


def test_connect_advertises_requested_tcp_mss_before_connecting() -> None:
    connection = MagicMock()
    address = ("192.0.2.1", 22)
    with (
        patch.object(
            tcp_mss_proxy.socket,
            "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address)
            ],
        ),
        patch.object(tcp_mss_proxy.socket, "socket", return_value=connection),
    ):
        result = tcp_mss_proxy.connect("example.invalid", 22, 1400)

    assert result is connection
    connection.setsockopt.assert_called_once_with(
        socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1400
    )
    assert connection.settimeout.call_args_list[0].args == (15.0,)
    connection.connect.assert_called_once_with(address)
    assert connection.settimeout.call_args_list[1].args == (None,)


@pytest.mark.parametrize("mss", [0, 255, 1461, 9000])
def test_main_rejects_unsafe_mss_values(mss: int) -> None:
    with pytest.raises(SystemExit, match="--mss must be"):
        tcp_mss_proxy.main(["192.0.2.1", "22", "--mss", str(mss)])


def test_parser_defaults_to_benchmarked_mss() -> None:
    args = tcp_mss_proxy.build_parser().parse_args(["192.0.2.1", "22"])
    assert args.mss == 1400
