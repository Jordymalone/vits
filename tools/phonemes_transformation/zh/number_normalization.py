#!/usr/bin/env Python
# coding=utf-8
import socket
import struct
import argparse
def askForService(language:str, chinese:str):
    '''
    數字正規化處理，將中文輸入中的阿拉伯數字跟符號，轉換成對應唸法的國字。
    可以選擇中文及台語兩種語言的轉換。
    中文的轉換結果統一為小寫數字國字。
    台語的轉換結果根據文白讀不同，給予不同大小寫的數字國字。
    EX: 
    壹(文) 一(白) 
    貳(文) 二(白)
    參(文) 三(白).....

    Params:
        language:   (str) ch or tl,ch=中文,tl=台語. Language will be normalze.
        chinese:    (str) chinese will be normalze.
    '''
    global HOST
    global PORT
    global TOKEN
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if len(language)==0 or len(chinese)==0:
            raise  ValueError ("Input language and chinese should not be empty!!!")
        sock.connect((HOST, PORT))
        msg = bytes(TOKEN + "@@@" + chinese + '@@@' + language, "utf-8")
        msg = struct.pack(">I", len(msg)) + msg        
        sock.sendall(msg)
        result=""
        while True:
            l = sock.recv(16382)
            if not l:
                break
            # result += l.decode(encoding="UTF-8")
            result += l.decode("UTF-8", 'ignore')
        
    finally:
        sock.close()
    return result

global HOST
global PORT
global TOKEN
HOST, PORT = "140.116.245.157", 2000
TOKEN = "mi2stts"

if __name__=='__main__':
    # ch = 中文, tl = 台語
    language = 'tl'
    data = "0939466777"
    # data = "班上第2位同學"
    # data = "04-5901987"
    # data = '20.5%'
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default=language, help='ch or tl, ch=中文, tl=台語. Language will be normalze.')
    parser.add_argument('--text', default=data, help='Text will be normalze.')
    args = parser.parse_args()
    result = askForService(language=args.language, chinese=args.text)
    print(result)
