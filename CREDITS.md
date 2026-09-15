# Credits

This project builds on other people's work. Thank you to everyone below.

## Pipecat Smart Turn (baseline model, reference code, data)

- **Code:** [github.com/pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn), BSD 2-Clause License, Copyright (c) 2024–2025, Daily. Contributors listed by the project: [Marcus](https://github.com/marcus-daily), [Eli](https://github.com/ebb351), [Mark](https://github.com/markbackman) and [Kwindla](https://github.com/kwindla).
  - Our model head (attention pooling + MLP classifier) follows the design in Smart Turn's `train.py`: `eot_lora/model.py`.
  - Our 8-second audio window and log-mel preprocessing follow Smart Turn's `inference.py` and `audio_utils.py`: `eot_lora/serving.py` and `eot_lora/audio.py`.
  - Our class-balanced loss follows `train.py`.
  - Smart Turn's source code is not included in this repository. Its license is reproduced at the end of this file for the adapted parts.
- **Model:** [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3), BSD 2-Clause. We use the Smart Turn v3.2 fp32 ONNX model as the evaluation baseline. It's downloaded at evaluation time and not redistributed here.
- **Data:** [pipecat-ai/smart-turn-data-v3.2-train](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) and [smart-turn-data-v3.2-test](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test). No license is declared on the dataset cards.
  - We used the English clips for training and evaluation. The data itself is not included here.
  - The 20 clips in `samples/` and the example audio embedded in `presentation.ipynb` are taken from the test set, some mixed with office noise.
  - The dataset card thanks these contributors of audio samples: the Pipecat team, [Liva AI](https://www.theliva.ai/), [Midcentury](https://www.midcentury.xyz/) and [MundoAI](https://mundoai.world/).
  - It also thanks these Freesound users for CC0 background-noise samples mixed into the dataset:
    [4team](https://freesound.org/people/4team/sounds/214995/),
    [tomhannen](https://freesound.org/people/tomhannen/sounds/698090/),
    [craigsmith](https://freesound.org/people/craigsmith/sounds/675073/),
    [mrmayo](https://freesound.org/people/mrmayo/sounds/351265/) ([2](https://freesound.org/people/mrmayo/sounds/351264/)),
    [martats](https://freesound.org/people/martats/sounds/156983/),
    [Duisterwho](https://freesound.org/people/Duisterwho/sounds/642196/),
    [el_boss](https://freesound.org/people/el_boss/sounds/635967/),
    [clivew](https://freesound.org/people/clivew/sounds/577104/),
    [dbspin](https://freesound.org/people/dbspin/sounds/396678/),
    [nikitralala](https://freesound.org/people/nikitralala/sounds/204406/),
    [outandaboutworcester](https://freesound.org/people/outandaboutworcester/sounds/670290/) ([2](https://freesound.org/people/outandaboutworcester/sounds/670289/), [3](https://freesound.org/people/outandaboutworcester/sounds/670291/)),
    [bengomori](https://freesound.org/people/bengomori/sounds/381697/),
    [wcarcary](https://freesound.org/people/wcarcary/sounds/155665/),
    [kentspublicdomain](https://freesound.org/people/kentspublicdomain/sounds/324668/),
    [Hmuryj_sound-Kirill_Rozhkov](https://freesound.org/people/Hmuryj_sound-Kirill_Rozhkov/sounds/616358/),
    [kyles](https://freesound.org/people/kyles/sounds/451024/) ([2](https://freesound.org/people/kyles/sounds/451105/), [3](https://freesound.org/people/kyles/sounds/454413/)),
    [edgardomoreno](https://freesound.org/people/edgardomoreno/sounds/450361/),
    [chimerical](https://freesound.org/people/chimerical/sounds/104279/),
    [lukabailen](https://freesound.org/people/lukabailen/sounds/723794/),
    [Vecera_999](https://freesound.org/people/Vecera_999/sounds/706805/),
    [Seamonstar](https://freesound.org/people/Seamonstar/sounds/718604/),
    [JW_Audio](https://freesound.org/people/JW_Audio/sounds/808266/),
    [Talitha5](https://freesound.org/people/Talitha5/sounds/509950/),
    [Ultra-Edward](https://freesound.org/people/Ultra-Edward/sounds/823831/),
    [seriousmedium33](https://freesound.org/people/seriousmedium33/sounds/543858/),
    [_vk](https://freesound.org/people/_vk/sounds/221848/),
    [holidayparade](https://freesound.org/people/holidayparade/sounds/278154/),
    [felix.blume](https://freesound.org/people/felix.blume/sounds/710089/),
    [Geoff-Bremner-Audio](https://freesound.org/people/Geoff-Bremner-Audio/sounds/705787/),
    [Sauron974](https://freesound.org/people/Sauron974/sounds/273624/),
    [Chelly01](https://freesound.org/people/Chelly01/sounds/541117/),
    [TRP](https://freesound.org/people/TRP/sounds/577495/),
    [servozero](https://freesound.org/people/servozero/sounds/636268/),
    [Garuda1982](https://freesound.org/people/Garuda1982/sounds/633121/),
    [joche.fernandez](https://freesound.org/people/joche.fernandez/sounds/460014/).

## Office noise recordings

- **"IT office 3"** (Pixabay ID 535970) and **"IT office 5"** (Pixabay ID 535971), by **thellywellyn** on [Pixabay](https://pixabay.com/), under the [Pixabay Content License](https://pixabay.com/service/license-summary/).
  - "IT office 3" is mixed into the training data of models B and C. "IT office 5" is used only for testing.
  - The original files are not redistributed here. The mixed clips in `samples/` and in the notebook contain them in altered form.

## Models and methods

- **Whisper:** OpenAI; Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022). We use the encoder of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) (Apache-2.0 on its model card; the original [openai/whisper](https://github.com/openai/whisper) release is MIT).
- **LoRA:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), through Hugging Face [PEFT](https://github.com/huggingface/peft).
- **WebRTC noise suppression:** the WebRTC audio processing module (the WebRTC project authors, BSD-3-Clause), used through the [LiveKit](https://github.com/livekit/python-sdks) Python SDK.

## Software

| Library | License |
|---|---|
| [PyTorch](https://pytorch.org) | BSD-3-Clause |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| [PEFT](https://github.com/huggingface/peft) | Apache-2.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 |
| [ONNX](https://github.com/onnx/onnx) | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT |
| [LiveKit Python SDK](https://github.com/livekit/python-sdks) | Apache-2.0 |
| [PyAV](https://github.com/PyAV-Org/PyAV) (bundles FFmpeg, LGPL) | BSD-3-Clause |
| [NumPy](https://numpy.org) | BSD-3-Clause |
| [pandas](https://pandas.pydata.org) | BSD-3-Clause |
| [PyArrow](https://arrow.apache.org) | Apache-2.0 |
| [scikit-learn](https://scikit-learn.org) | BSD-3-Clause |
| [FastAPI](https://fastapi.tiangolo.com) | MIT |
| [Uvicorn](https://www.uvicorn.org) | BSD-3-Clause |
| [Pydantic](https://docs.pydantic.dev) | MIT |
| [HTTPX](https://www.python-httpx.org) | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org) | Matplotlib License (PSF-based) |
| [JupyterLab](https://jupyter.org), nbformat, nbclient | BSD-3-Clause |

Code, experiments and write-up were developed with the help of [Claude Code](https://claude.com/claude-code) (Anthropic).

## Smart Turn license (for the adapted parts)

```
BSD 2-Clause License

Copyright (c) 2024–2025, Daily

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
