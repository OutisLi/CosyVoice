import sys
import torch
import librosa

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio # type: ignore


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    max_val = 0.8
    speech, _ = librosa.effects.trim(speech, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
# prompt_speech_16k = load_wav('./asset/sample.wav', 16000)
prompt_speech_16k = postprocess(load_wav("./asset/sample.wav", 16000))
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        "那我今天就早一点睡，但其实我今天还在纠结那个选课，然后因为我发现了一门很好很好的课，就是跟我的专业没什么关系，就是比数字法治还要好，然后我在纠结要不要上那个，还是就上今天这一节好了。",
        "那我今天就早一点睡，但其实我今天还在纠结那个选课，然后因为我发现了一门很好很好的课，就是跟我的专业没什么关系，就是比数字法治还要好，然后我在纠结要不要上那个，还是就上今天这一节好了。",
        prompt_speech_16k,
        stream=False,
    )
):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# save zero_shot spk for future usage
assert (
    cosyvoice.add_zero_shot_spk(
        "那我今天就早一点睡，但其实我今天还在纠结那个选课，然后因为我发现了一门很好很好的课，就是跟我的专业没什么关系，就是比数字法治还要好，然后我在纠结要不要上那个，还是就上今天这一节好了。",
        prompt_speech_16k,
        "my_spk",
    )
    is True
)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_spk', stream=False)):
    torchaudio.save("zero_shot_spk_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate)
cosyvoice.save_spkinfo()

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
