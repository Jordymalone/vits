#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
Tai-lo sandhi + CTL 轉換整合版 client

流程：
1. tailo -> sandhi (port 2003)
2. sandhi -> CTL (port 2004)

使用方式：
  python sandhi2ctl.py --text "tsit8-e5 lang5"
  python sandhi2ctl.py --text "tsit8-e5 lang5" --stage sandhi
  python sandhi2ctl.py --text "tsit8-e5 lang5" --stage tl2ctl
"""

import socket
import struct
import argparse

# ===================== 設定區 =====================

TOKEN = "mi2stts"

SANDHI_HOST = "140.116.245.157"
SANDHI_PORT = 2003

TL2CTL_HOST = "140.116.245.157"
TL2CTL_PORT = 2004

# ===================== 共用底層函式 =====================

def _call_service(host: str, port: int, token: str, text: str, timeout: float = 10.0) -> str:
    """
    與指定的 TCP 服務通訊。

    Protocol:
        msg = struct.pack(">I", len(payload)) + payload
        payload = (token + "@@@" + text).encode("utf-8")
    收到的結果以 UTF-8 decode 回傳為字串。
    """
    if not text:
        raise ValueError("Input text should not be empty!")

    payload = f"{token}@@@{text}".encode("utf-8")
    msg = struct.pack(">I", len(payload)) + payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect((host, port))
        sock.sendall(msg)

        chunks = []
        while True:
            l = sock.recv(8192)
            if not l:
                break
            chunks.append(l)

        result = b"".join(chunks).decode("utf-8")
        return result
    finally:
        try:
            sock.close()
        except Exception:
            pass

# ===================== 高階 API =====================

def tailo_to_sandhi(tai_luo: str) -> str:
    """
    將輸入的台羅拼音進行轉調（sandhi）。

    若句子的台羅拼音不是從 ch2tl 的 API 產生，
    麻煩請將獨立語意的詞彙用 hyphen(-) 連接起來。
    """
    return _call_service(SANDHI_HOST, SANDHI_PORT, TOKEN, tai_luo)


def sandhi_to_ctl(sandhi_tl: str) -> str:
    """
    將 sandhi 後的台羅拼音轉換成實驗室使用的 CTL 拼音。
    """
    return _call_service(TL2CTL_HOST, TL2CTL_PORT, TOKEN, sandhi_tl)

# ===================== CLI 介面 =====================

def main():
    parser = argparse.ArgumentParser(
        description="Tai-lo sandhi + CTL pipeline client"
    )
    parser.add_argument(
        '--text',
        type=str,
        default="gua2 si7 sing5-kong1 tai7-hak8 hak8-sing1",
        help='原始台羅拼音（未 sandhi）。'
    )
    parser.add_argument(
        '--stage',
        choices=['sandhi', 'tl2ctl', 'both'],
        default='both',
        help=(
            "sandhi  : 只執行 tailo -> sandhi，輸出 sandhi 結果\n"
            "tl2ctl  : 只執行 sandhi -> CTL，假設輸入 text 已是 sandhi 結果\n"
            "both    : 先 tailo -> sandhi，再 sandhi -> CTL（預設）"
        )
    )
    args = parser.parse_args()

    text = args.text

    if args.stage == 'sandhi':
        sandhi = tailo_to_sandhi(text)
        print("=== Sandhi 結果 ===")
        print(sandhi)

    elif args.stage == 'tl2ctl':
        ctl = sandhi_to_ctl(text)
        print("=== CTL 結果 ===")
        print(ctl)

    else:  # both
        sandhi = tailo_to_sandhi(text)
        print("=== Sandhi 結果 ===")
        print(sandhi)

        ctl = sandhi_to_ctl(sandhi)
        print("=== CTL 結果 ===")
        print(ctl)


if __name__ == '__main__':
    main()