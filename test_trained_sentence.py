import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
from scipy.io.wavfile import write

print(f"符號表長度: {len(symbols)}")

# 載入配置
hps = utils.get_hparams_from_file('configs/hakka_hm_model.json')

# 載入模型
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()

_ = utils.load_checkpoint('logs/hakka_hm_model/G_302000.pth', net_g, None)
_ = net_g.eval()

# 使用訓練資料中的第一句音素（複製自訓練資料）
phonemes = "tsr u21 ph oon23 sr i23 tsc iet28 ， th eu23 n a23 m oo28 ooi25 th ak28 h i21 l ooi23 。"

print(f"測試音素: {phonemes}")

# 轉換成序列
try:
    stn_tst = cleaned_text_to_sequence(phonemes, 'HAK')
    print(f"✅ 音素序列長度: {len(stn_tst)}")
    
    # 生成音訊
    with torch.no_grad():
        x_tst = torch.LongTensor([stn_tst]).cuda()
        x_tst_lengths = torch.LongTensor([len(stn_tst)]).cuda()
        
        print("正在生成音訊...")
        audio = net_g.infer(x_tst, x_tst_lengths, 
                            noise_scale=0.667, 
                            noise_scale_w=0.8, 
                            length_scale=1.0)[0][0,0].data.cpu().float().numpy()
    
    # 保存
    output_file = 'hakka_epoch1393_test.wav'
    write(output_file, hps.data.sampling_rate, audio)
    
    print(f"\n✅ 成功！")
    print(f"   輸出檔案: {output_file}")
    print(f"   取樣率: {hps.data.sampling_rate} Hz")
    print(f"   音訊長度: {len(audio)/hps.data.sampling_rate:.2f} 秒")
    print(f"\n播放測試: aplay {output_file}")
    
except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
