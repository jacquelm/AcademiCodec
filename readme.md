# Trainer for SpeechTokenizer

#### Abstract
Current speech large language models build upon discrete speech representations, which can be categorized into semantic tokens and acoustic tokens. However, existing speech tokens are not specifically designed for speech language modeling. To assess the suitability of speech tokens for building speech language models, we established the first benchmark, SLMTokBench. Our results indicate that neither semantic nor acoustic tokens are ideal for this purpose. Therefore, we propose SpeechTokenizer, a unified speech tokenizer for speech large language models. SpeechTokenizer adopts the Encoder-Decoder architecture with residual vector quantization (RVQ). Unifying semantic and acoustic tokens, SpeechTokenizer disentangles different aspects of speech information hierarchically across different RVQ layers. Furthermore, We construct a Unified Speech Language Model (USLM) leveraging SpeechTokenizer. Experiments show that SpeechTokenizer performs comparably to EnCodec in speech reconstruction and demonstrates strong performance on the SLMTokBench benchmark. Also, USLM outperforms VALL-E in zero-shot Text-to-Speech tasks. Code and models are available at this https URL.

# Train your own model
```commandline
bash <path_to_>/SpeechTokenizer_trainer/academicodec/models/speechtokenzier/train.sh
```

## Data preparation
Just prepare your audio data in one folder. Make sure the sample rate is right.
HuBERT and kmeans for the distillation of the SpeechTokenizer need to be prepared.
HuBERT and kmeans are described in details [here]
(https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm).

```commandline
bash <path_to_>/SpeechTokenizer_trainer/academicodec/models/speechtokenizer/extract_feature.sh
```

## Usage
### Model storage
| Model| Dataset |Discription|
|:----|:----:|:----|
|[speechtokenizer_hubert_avg](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg)|LibriSpeech|Adopt average representation across all HuBERT layers as semantic teacher |
### load model
```python
from speechtokenizer import SpeechTokenizer

config_path = '/path/config.json'
ckpt_path = '/path/SpeechTokenizer.pt'
model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
model.eval()
```

### Extracting discrete representations
```python
import torchaudio
import torch

# Load and pre-process speech waveform
wav, sr = torchaudio.load('<SPEECH_FILE_PATH>')

# monophonic checking
if wav.shape(0) > 1:
    wav = wav[:1,;]

if sr != model.sample_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

wav = wav.unsqueeze(0)

# Extract discrete codes from SpeechTokenizer
with torch.no_grad():
    codes = model.encode(wav) # codes: (n_q, B, T)

RVQ_1 = codes[:1, :, :] # Contain content info, can be considered as semantic tokens
RVQ_supplement = codes[1:, :, :] # Contain timbre info, complete info lost by the first quantizer
```

### Decoding discrete representations
```python
# Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
wav = model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))

# Decoding from RVQ-i:j tokens from the ith quantizers to the jth quantizers
wav = model.decode(codes[i: (j + 1)], st=i) 
```

## Points to note.
The original SpeechTokenizer appears to have been trained at 16 kHz.
This is because the HuBERT and Kmeans used as teachers are trained at 16 kHz in the publicly available version.
If you want to train above 16 kHz, you need HuBERT and HuBERT kmeans trained above 16 kHz.
Detailed training instructions on that can be found [here]
(https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm).


## Acknowledgements
This implementation uses parts of the code from the following Github repos:
https://github.com/ZhangXInFD/SpeechTokenizer <br>
https://github.com/yangdongchao/AcademiCodec <br>

## Citation
If you use this code or result in your paper, please cite our work as:
```Tex
@misc{zhang2023speechtokenizer,
      title={SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models}, 
      author={Xin Zhang and Dong Zhang and Shimin Li and Yaqian Zhou and Xipeng Qiu},
      year={2023},
      eprint={2308.16692},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```