# AI Voice Clone (STT - LLM - TTS)

## Introduction

Assistant vocal qui vous écoute et qui vous répond avec une voix clonée (de votre choix).

1. Écoute : Enregistre l'audio du micro.
2. Transcription (Whisper) : Utilise whisper-cli ou la bibliothèque Python pour transformer l'audio en texte (STT).
3. Génération (Qwen) : Envoie le texte à llama.cpp pour obtenir une réponse (LLM).
4. Synthèse (pocket-tts) : Transforme la réponse en audio pour la restituer.

- Application réalisée sur une VM Debian 13.
- Périphériques: micro et écouteurs.

---

## Installation

- Création d'un environnement virtuel:

```
python3 -m venv myenv
source /venv/bin/activate
```

---

## LLama install (GGUF)

- Le serveur

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
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

## Qwen install (GGUF)

- La version de Qwen ci-dessous prend en charge le français.

```
# Meilleure version multi (Qwen3 à tester et Qwen3.5 payante) 
qwen2.5-1.5b-instruct-q4_k_m.gguf (instruct=modèle conversationnels)

# Télécharge le fichier spécifique dans le dossier ./models
hf download ggml/qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models
```

---

## Sherpa-onnx install (optionnel)

- Personnellement, pas très convaincu...

```
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

# Extraire l'archive dans le dossier ./models:
tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

# Supprimer l'archive pour économiser de l'espace
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
```

---

`sudo apt install pulseaudio pulseaudio-utils`

## Installations avec pip

`pip install sherpa-onnx`

`pip install pocket-tts`

`pip install speechrecognition pyaudio requests numpy soundfile`

---

## Générer une voix clonée avec pocket-tts

Créer un fichier `ma_voix.wav` ou utiliser un fichier `.wav`.

Génération d'un fichier `.safetensors`:

`pocket-tts export-voice ma_voix.wav voix_clonee.safetensors`

La latence est réduite en utilisant la commande suivante avec `.safetensors`:

`pocket-tts generate --voice clone.safetensors --text "Bonjour, c'est ma voix clonée !"`

Et de réutiliser `.safetensors` dans le script python `assistant.py` (réduction de latence).

---

## HuggingFace

- Quelques commandes:

`hf auth login`

`hf auth whoami`

- Télécharge le fichier spécifique dans le dossier `./models`:

`hf download ggml/qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ./models`

- Supprimer le cache avec hf

`hf cache ls` (lister le cache)

`hf cache rm model/Qwen/Qwen2-0.5B-Instruct` (delete le model)

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

## :snake: Start python script

`python3 assistant.py`

`python3 assistant.py --voice ./tester3.wav`

`python3 assistant.py 2>/dev/null` (NNPACK's bug from PyTorch hidden)

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

- under dev (je travaille avec l'option --quantize)

---

Enjoy it ! :koala:
