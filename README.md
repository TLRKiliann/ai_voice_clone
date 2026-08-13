# llama - whisper - qwen - pocket-tts

# générer une voix clonée avec pocket-tts

`pocket-tts export-voice ma_voix.wav voix_clonee.safetensors`

---

## LLama install

```
git clone https://github.com/ggerganov/llama.cpp.git
cd nom
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

---

## Qwen install

```
# First install of Qwen
wget https://huggingface.co/Dev8709/Qwen2.5-0.5B-Q4_K_M-GGUF/resolve/main/qwen2.5-0.5b-q4_k_m.gguf
./build/bin/llama-server -m ./models/qwen2.5-0.5b-q4_k_m.gguf -c 2048 --host 0.0.0.0 --port 8080

# Best Qwen's version for french
qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

## Whisper install

```
cd ~
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build && cmake --build build -j --config Release
cd ~/whisper.cpp
./models/download-ggml-model.sh base

./build/bin/whisper-cli -m models/ggml-base.bin --help
```

---

## Sherpa-onnx install

```
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

# Extraire l'archive
tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

# Supprimer l'archive pour économiser de l'espace
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
```

`pip install sherpa-onnx`

`pip install pocket-tts`

---

```
/home/tester/
├── llama.cpp/           ← Pour Qwen (LLM)
│   ├── build/
│   │   └── bin/
│   │       └── llama-server
│   └── ...
├── whisper.cpp/         ← Pour la reconnaissance vocale
│   ├── build/
│   │   └── bin/
│   │       └── whisper-cli
│   ├── models/
│   │   └── ggml-base.bin
│   └── ...
├── CrispASR/            ← Pour Pocket-TTS
│   ├── build/
│   │   └── bin/
│   │       └── crispasr-cli
│   └── ...
├── models/              ← Vos modèles GGUF (partagés)
│   ├── qwen2.5-0.5b-q4_k_m.gguf
│   └── pocket-tts-french_24l-q8_0.gguf
└── assistant.py         ← Votre script Python d'intégration
```

---

## Start server

```
cd ./llama.cpp

./build/bin/llama-server -m ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 2048 --host 0.0.0.0 --port 8080
```

## Start in front

`python3 assistant.py --voice ./tester3.wav`

---

## Meilleure version avec fr

`qwen2.5-1.5b-instruct-q4_k_m.gguf`

`pip install speechrecognition pyaudio requests numpy soundfile`

`sudo apt install pulseaudio pulseaudio-utils`

---

## Sound tests

```
# list audio drivers
arecord -l

# sherpa-onnx 16000Hz
arecord -d 5 -f cd -r 16000 -c 1 ma_voix.wav

# pocket-tts 24000Hz
arecord -d 5 -f cd -r 24000 -c 1 ma_voix.wav

./whisper.cpp/build/bin/whisper-cli -m ~/whisper.cpp/models/ggml-base.bin -f test.wav -l fr
```