# !/usr/bin/env python
# - - coding: utf-8 - -**

import socket
import struct
import argparse
import json

def askForService(text:str, accent:str, direction:str):
    '''
        將輸入的中文轉換成客語，或客語漢字轉為中文。\
        todo: 若輸入為客語數字調，則輸出亦為客語數字調，可將符號調客語轉成數字調客語。\
            Params:
            text:       (str) Text will be translate from Chinese to Hakka.
            accent: 客語腔調 (可用四縣(hedusi)或海陸(hedusai)腔)
            direction: 中翻客(ch2hk)或客翻中(hk2ch)
            2023/7/30: direction新增客語漢字到拼音(hkji2pin)
        output: dict形式
        out_dict["hakkaTL"] : 翻譯客語結果 (數字調)
        out_dict["interCH"] : 翻譯客語結果 (客語漢字)
        out_dict["hakkaTRN"] : 翻譯客語結果 (數字調) 轉成trn
        
    '''
    global HOST
    global PORT
    global TOKEN
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        if len(text)==0:
            raise  ValueError ("Input text should not be empty!!!")
        sock.connect((HOST, PORT))
        msg = bytes(TOKEN + "@@@" + text + "@@@" + accent + "@@@" + direction, "utf-8")
        msg = struct.pack(">I", len(msg)) + msg
        sock.sendall(msg)
        result = ""
        while True:
            l = sock.recv(8192)
            if not l:
                break
            result += l.decode(encoding="UTF-8")
            TLresult = json.loads(result)
    except Exception as e:
        print(e)
        return {"hakkaTRN": "Exceptions occurs"}
            
    finally:
        sock.close()
        
    return TLresult

global HOST
global PORT
global TOKEN
HOST, PORT = "140.116.245.157", 30005
TOKEN = "mi2stts"

if __name__=='__main__':
    text = "我先去喝水"
    text2 = "景氣毋好，生理失敗背債个人，緊來緊多。"
    text3 = "豆腐蘸豆油食盡合味"
    text = '晚上去散步'
    text = '有兜人主張打胎合法 毋過有兜人極力反對'
    text = '人之初，性本善。' 
    text = 'n11 dui55 nai55 deu24 siid5 vud5 voi55 go55 men31'
    text = '你好'
    text = 'he55 do55 nga24 vug2 ha24 fu55 kiun24 ziin24 so31 ge55 i24 sen24 koi24 ge55'
    text = '請問你這下有哪位感覺著毋鬆爽'
    text = 'qiang31 mun55 n11 ia31 ha55 iu24 nai55 vi55 gam31 gog2 do31 m11 sung24 song3'
    text = '藥房愛仰般行'
    text = 'iog5 fong11 oi55 ngiong31 ban24 hang11'
    text = '請問醫院有哪幾科門診'
    text = '建議你掛泌尿科門診'
    text = '最近無'
    text = '該你食東西愛注意控制正好'
    text = 'nai55 vi55 gam31 gog2 do31 iu24 mun55 ti11'
    text = '客家本聲'
    text = 'iu24 bi55 gied2 oi55 do24 siid5 go24 xien24 vi11 ge55 siid5 vud5 do24 lim24 sui31 han11 go55 siid2 dong24 ge55 iun55 tung55'
    # text = 'hag24 ga31 bun55 sed2'
    accent = 'hedusi' # 四縣
    # accent = 'hedusai' # 海陸
    direction1 = "ch2hk"            # 中文 -> 客語漢字 (看 interCH 欄位)
    direction2 = "hakji2pin"        # 客語漢字 -> 拼音 (accent, 看 hakkaTRN 欄位)
    direction3 = "hakpin2trn"       # 客語拼音 -> TRN (看 trn 欄位)
    # direction2 = "hk2ch"
    # direction3 = "hkji2pin"
    direction4 = "hkji2trn"
    direction5 = "hakka_pinyin"
    direction6 = "hakpin2trn"
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=text, help='Text will be translate from Chinese to Hakka.')
    parser.add_argument('--accent', default=accent, help='Select your hakka accent (available in hedusi and hedusai).')
    parser.add_argument('--direction', default=direction3, help='TL direction includes ch2hk and hk2ch.')
    args = parser.parse_args()
    #args = parser.parse_args()
    
    result = askForService(text=args.text, accent=args.accent, direction = args.direction)
    print(result)
    # if (result['hakkaTRN'] == 'Exceptions occurs'):
    #     print(f'Error: {result["hakkaTRN"]}')

    