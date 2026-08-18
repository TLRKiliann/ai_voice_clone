# AI Voice Clone (STT - LLM - TTS)

## :boom: Introduction

Assistant vocal qui vous écoute et qui vous répond avec une voix clonée (de votre choix).

1. Écoute : Enregistre l'audio du micro.
2. Transcription (Whisper) : Utilise whisper-cli ou la bibliothèque Python pour transformer l'audio en texte (STT).
3. Génération (Qwen) : Envoie le texte à llama.cpp pour obtenir une réponse (LLM).
4. Synthèse (pocket-tts) : Transforme la réponse en audio pour la restituer.

- Application réalisée sur une VM Debian 13.
- Périphériques: micro et écouteurs.

---

## :scroll: Installation

- Prérequis (cmd pour faire des tests à la fin de cette page)

```
sudo apt update
sudo apt install build-essential cmake
sudo apt install pulseaudio pulseaudio-utils ffmpeg
```

- Création d'un environnement virtuel:

```
python3 -m venv myenv
source myvenv/bin/activate
```

- Créer le dossier models pour Qwen:

`mkdir models`

---

## :busts_in_silhouette: LLama install (GGUF)

- Le serveur

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# Display options 
./build/bin/llama-server --help
```

---

## :microphone: Whisper install (ggml => GGUF) :pen:

```
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build && cmake --build build -j --config Release
./models/download-ggml-model.sh base

# Display options 
./build/bin/whisper-cli -m models/ggml-base.bin --help

# Reconnaissance vocale pour la retranscription par écrit
# (on en a pas besoin, la cmd est intégrée dans le code assistant.py)
./whisper-cli -m models/ggml-base.bin -f votre_fichier.wav -l fr
```

---

## :envelope: Qwen install (GGUF)

- La version de Qwen ci-dessous prend en charge le français.

```
# Meilleure version multi (Qwen3 à tester et Qwen3.5 payante) 
qwen2.5-1.5b-instruct-q4_k_m.gguf (instruct=modèle conversationnels)

# Download
wget -O ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

- Dans le navigateur on peut utiliser Qwen pour interagir avec: 

`http://localhost:8080`

---

## :nazar_amulet: Installations avec pip

`pip install pocket-tts[quantize]`

`pip install speechrecognition pyaudio requests numpy soundfile`

---

## :sound: Générer une voix clonée avec pocket-tts

Créer un fichier `ma_voix.wav` ou utiliser un fichier `.wav`.

Génération d'un fichier `.safetensors`:

`pocket-tts export-voice ma_voix.wav clone.safetensors`

La latence est réduite en utilisant la commande suivante avec `.safetensors`:

`pocket-tts generate --voice clone.safetensors --text "Bonjour, c'est ma voix clonée !"`

Et de réutiliser `.safetensors` dans le script python `assistant.py` (réduction de latence).

---

## :hugs: HuggingFace

- Quelques commandes:

`hf auth login`

`hf auth whoami`

- Télécharge le fichier spécifique dans le dossier `./models`:

`hf download ggml/qwen2.5-0.5b-example.gguf --local-dir ./models`

- Supprimer le cache avec hf

`hf cache ls` (lister le cache)

`hf cache rm model/Qwen/Qwen2-0.5B-Instruct` (delete le model)

---

## :atom: Start server

```
cd ./llama.cpp

./build/bin/llama-server -m ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 2048 --host 0.0.0.0 --port 8080
```

- Options

```
-c 2048		Équilibre entre performance et qualité
-c 32768	Idéal pour de longs documents ou des conversations suivies
```

---

## :snake: Start python script

- Différentes options:

`python3 assistant.py` (standard)

`python3 assistant.py --duration 10` (10 secondes pour parler)

`python3 assistant.py --voice ./tester3.wav` (choix du fichier .wav ou .safetensors)

`python3 assistant.py 2>/dev/null` (cacher les erreur NNPACK de PyTorch)

- Exemple: `python3 assistant.py --duration 10 2>/dev/null`

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

## :sound: Sound tests

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

## :gift: Extra

- Model de raisonnement encore plus poussé avec DeepSeek:

```
cd llama.cpp

wget -O ../models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf

./build/bin/llama-server -m ../models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf -c 2048 --host 0.0.0.0 --port 8080
```

---

Enjoy it ! :koala:
