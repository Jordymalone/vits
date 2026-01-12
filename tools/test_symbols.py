#!/usr/bin/env python3
"""
測試 symbols.py 是否正確載入情緒音素
"""
import sys
import os

# 加入專案根目錄
sys.path.insert(0, '/mnt/Linux_DATA/synthesis/model/vits')

from text.symbols import symbols, _symbol_to_id

print(f'✅ 成功載入 symbols，總數: {len(symbols)}')
print(f'前 10 個符號: {symbols[:10]}')
print(f'後 5 個符號: {symbols[-5:]}')

# 測試原始錯誤訊息中的音素
test_phonemes = ['uo13', 'm1', 'en15', 'k1', 'e13', 'i13', 'q1', 'v14', 'd1', 'iao14', 'v12', 'sp']
print(f'\n測試原始錯誤中的音素是否存在:')
all_found = True
for ph in test_phonemes:
    if ph in _symbol_to_id:
        print(f'  ✅ {ph} -> ID {_symbol_to_id[ph]}')
    else:
        print(f'  ❌ {ph} 不存在')
        all_found = False

if all_found:
    print(f'\n🎉 所有測試音素都存在於 symbol_to_id 映射中！')
else:
    print(f'\n⚠️ 部分音素缺失，請檢查 emotion_phones.txt')
