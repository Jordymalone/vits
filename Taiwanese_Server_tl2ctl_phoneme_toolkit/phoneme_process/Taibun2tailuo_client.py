# !/usr/bin/env python
# _*_coding:utf-8_*_

import socket
import struct
import argparse
def askForService(tai_luo:str):
    '''
    將輸入的台羅拼音進行轉調
    若句子的台羅拼音不是從 ch2tl的api產生的
    麻煩請將獨立語意的詞彙用hyphen的符號(-)連接起來
    Params:
        tai_luo    :(str) Tai-luo will be sandhi.
    '''
    global HOST
    global PORT
    global TOKEN
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if len(tai_luo)==0:
            raise  ValueError ("Input text should not be empty!!!")
        sock.connect((HOST, PORT))
        msg = bytes(tai_luo, "utf-8")
        msg = struct.pack(">I", len(msg)) + msg
        sock.sendall(msg)
        result=""
        while True:
            l = sock.recv(8192)
            if not l:
                break
            result += l.decode(encoding="UTF-8")
    finally:
        sock.close()
    return result

global HOST
global PORT
global TOKEN
HOST, PORT = "140.116.245.157", 30331
TOKEN = "mi2stts"

if __name__=='__main__':
    
    data = "請到兒童病院邊仔的等車遐坐車"
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=data, help='Text will be sandhi.')
    args = parser.parse_args()
    result = askForService(tai_luo=args.text)
    print(result)