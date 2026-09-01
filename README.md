# AI Voice Clone (STT - LLM - TTS)

## :boom: Introduction

FR
Assistant vocal qui vous écoute et qui vous répond avec une voix clonée (de votre choix).

EN
A voice assistant that listens to you and responds with a cloned voice (of your choice).

(fr)
1. Écoute : Enregistre l'audio du micro.
2. Transcription (Whisper) : Utilise whisper-cli ou la bibliothèque Python pour transformer l'audio en texte (STT).
3. Génération (Qwen) : Envoie le texte à llama.cpp pour obtenir une réponse (LLM).
4. Synthèse (pocket-tts) : Transforme la réponse en audio pour la restituer.

(en)
1. Listening: Records audio from the microphone.
2. Transcription (Whisper): Uses whisper-cli or the Python library to convert audio to text (STT).
3. Generation (Qwen): Sends the text to llama.cpp to obtain a response (LLM).
4. Synthesis (pocket-tts): Converts the response into audio for playback.

(fr)
- Application réalisée sur une VM Debian 13.
- Périphériques: micro et écouteurs.

(en)
- Application running on a Debian 13 VM.
- Peripherals: microphone and headphones.

---

## :scroll: Installation

- Prerequisites (cmd for running tests at the end of this page)

```
sudo apt update
sudo apt install build-essential cmake
sudo apt install pulseaudio pulseaudio-utils ffmpeg
```

- Create virtuel environment:

```
python3 -m venv myenv
source myvenv/bin/activate
```

- Create the folder models for Qwen:

`mkdir models`

---

## :busts_in_silhouette: LLama install (GGUF)

- Server

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

(fr)
- Dans le navigateur on peut utiliser Qwen pour interagir avec: 

(en)
- In the browser, you can use Qwen to interact with:

`http://localhost:8080`

---

## :nazar_amulet: Installations avec pip

⚠️ Version of PyTorch 2.10+ required ⚠️

- On Raspberry Pi 5, install torch & torchao with the following cmd:

`pip install torch torchao --index-url https://download.pytorch.org/whl/cpu`

- and then, run this cmd:

`pip install pocket-tts[quantize]`

`pip install speechrecognition pyaudio requests numpy soundfile`

---

## :sound: Générer une voix clonée avec pocket-tts

(fr)
Créer un fichier `ma_voix.wav` ou utiliser un fichier `.wav`.

Génération d'un fichier `.safetensors`:

(en)
Create a file named `my_voice.wav` or use an existing `.wav` file.

Generating a `.safetensors` file:

`pocket-tts export-voice ma_voix.wav clone.safetensors`

(fr)
La latence est réduite en utilisant la commande suivante avec `.safetensors`:

(en)
Latency is reduced by using the following command with `.safetensors`:

`pocket-tts generate --quantize --voice clone.safetensors --text "Bonjour, c'est ma voix clonée !"`

```
pocket-tts generate --quantize --language french_24l --text "text to speech..." \
    --voice test.safetensors --temperature 0.7 --lsd-decode-steps 5 --output-path new.wav
```

(fr)
Et de réutiliser `.safetensors` dans le script python `assistant.py` (réduction de latence).

(en)
And reuse `.safetensors` in the Python script `assistant.py` (to reduce latency).


---

## :hugs: HuggingFace

- Some commands:

`hf auth login`

`hf auth whoami`

- Download specific file onto folder `./models`:

`hf download ggml/qwen2.5-0.5b-example.gguf --local-dir ./models`

- Delete cache with `hf`:

`hf cache ls` (list the cache)

`hf cache rm model/Qwen/Qwen2-0.5B-Instruct` (delete model)

---

## :atom: Start server

- Prioritize faster-whisper on ARM64 architecture, such as Raspberry Pi 5.

Fast on CPU & doesn't require NPU (AI HAT+ 2).

- For PC:

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

- Script with options:

`python3 assistant.py` (standard)

`python3 assistant.py --duration 10` (10 secondes for speaking)

`python3 assistant.py --voice ./tester3.wav` (option's file .wav ou .safetensors)

`python3 assistant.py 2>/dev/null` (hide errors NNPACK of PyTorch)

- Example: `python3 assistant.py --duration 10 2>/dev/null`

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

- An even more advanced reasoning model with DeepSeek

```
cd llama.cpp

wget -O ../models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf

./build/bin/llama-server -m ../models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf -c 2048 --host 0.0.0.0 --port 8080
```

---

Enjoy it ! :koala:
