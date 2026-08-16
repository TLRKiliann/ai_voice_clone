# AI Voice Clone

## Introduction

Une app qui nous répond avec notre propre voix en français, avec une latence
de quelques secondes, mais j'y travaille encore (under dev)

1. Écoute : Enregistre l'audio du micro.
2. Transcription (Whisper) : Utilise whisper-cli ou la bibliothèque Python pour transformer l'audio en texte.
3. Génération (Qwen) : Envoie le texte à llama.cpp pour obtenir une réponse.
4. Synthèse (pocket-tts) : Transforme la réponse en audio pour la restituer.

---

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

- Instruct = modèles conversationnels.
- La version ci-dessous prend en charge le français.

```
# Best version multi
qwen2.5-1.5b-instruct-q4_k_m.gguf

# Télécharge le fichier spécifique dans le dossier ./models
hf download ggml/qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models
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

---

## Installations avec pip

`pip install sherpa-onnx`

`pip install pocket-tts`

`pip install speechrecognition pyaudio requests numpy soundfile`

---

## HuggingFace

Télécharge le fichier spécifique dans le dossier ./models:

`hf download ggml/qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models`

## Remove cache with hf

`hf cache ls` (lister le cache)

`hf cache rm model/Qwen/Qwen2-0.5B-Instruct` (delete le model)

---

```
my_project/
├── assistant.py           # Votre script
├── llama.cpp/            # Dossier du serveur LLM
│   └── build/bin/llama-server
├── models/               # ← DOSSIER MODELS À LA RACINE
│   ├── qwen2.5-1.5b-instruct-q4_k_m.gguf  # ← Le modèle est ici !
│   └── ...
├── whisper.cpp/          # Dossier Whisper
│   ├── build/bin/whisper-cli
│   └── models/ggml-base.bin  # ← Modèle Whisper dans son propre dossier
└── test.safetensors      # Référence vocale
```

---

## Start server

```
cd ./llama.cpp

# Display options 
./build/bin/llama-server --help

./build/bin/llama-server -m ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 2048 --host 0.0.0.0 --port 8080
```

- Options

```
-c 2048		Équilibre entre performance et qualité
-c 32768	Idéal pour de longs documents ou des conversations suivies
```

---

## Start python script

`python3 assistant.py`

`python3 assistant.py --voice ./tester3.wav`

`python3 assistant.py 2>/dev/null` (NNPACK from PyTorch hidden)

---

## Meilleure version avec fr

`qwen2.5-1.5b-instruct-q4_k_m.gguf`
`Qwen3.5-0.8B.GGUF`

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

---

## Python3 Script

"max_token" => en cours de dépréciation => "max_completion_tokens": 200 with new version.

```
{
  "messages": [
    { "role": "system", "content": "Vous êtes un assistant utile." },
    { "role": "user", "content": "Bonjour, comment ça va ?" }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 200,
  "stream": true
}
```

---

Enjoy it ! :koala:
