# 

## Présentation

1. Écoute : Enregistre l'audio du micro.
2. Transcription (Whisper) : Utilise whisper-cli ou la bibliothèque Python pour transformer l'audio en texte.
3. Génération (Qwen) : Envoie le texte à llama.cpp pour obtenir une réponse.
4. Synthèse (pocket-tts) : Transforme la réponse en audio pour la restituer.

## Générer une voix clonée avec pocket-tts

`pocket-tts export-voice ma_voix.wav voix_clonee.safetensors`

Ensuite, il est préférable d'utiliser la commande suivante pour gagner du temps avec `.safetensors` :

`pocket-tts generate --voice clone.safetensors --text "Bonjour, ceci est ma voix clonée !"`

---

## LLama install (GGUF)

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

---

## Qwen install (GGUF)

```
# Best version multi
qwen2.5-1.5b-instruct-q4_k_m.gguf

# Télécharge le fichier spécifique dans le dossier ./models
huggingface-cli download jc-builds/Qwen2.5-1.5B-Instruct-Q4_K_M-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models
```

---

## Whisper install (ggml => GGUF)

```
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
