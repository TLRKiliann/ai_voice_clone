#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASSISTANT VOCAL STREAMING + DIAGNOSTICS

Pipeline :

Micro
  ↓
arecord
  ↓
Whisper.cpp
  ↓
llama.cpp /v1/chat/completions
  ↓
LLM streaming
  ↓
TextChunker
  ↓
Queue TTS
  ↓
Pocket-TTS generate_audio_stream()
  ↓
Queue audio
  ↓
aplay
  ↓
Haut-parleur

Le programme mesure notamment :

- temps avant le premier token LLM
- vitesse du LLM
- temps avant le premier chunk audio TTS
- vitesse réelle de Pocket-TTS
- durée audio produite
- nombre de chunks audio
- état de la queue audio

Cela permet de déterminer précisément
l'origine des coupures audio.
"""

import os
import json
import re
import time
import queue
import argparse
import threading
import subprocess
import tempfile

import requests
import numpy as np

from pocket_tts import TTSModel


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # Whisper.cpp
    # --------------------------------------------------------

    "whisper_cli":
        "./whisper.cpp/build/bin/whisper-cli",

    "whisper_model":
        "./whisper.cpp/models/ggml-base.bin",

    # --------------------------------------------------------
    # llama.cpp
    # --------------------------------------------------------

    "llm_url":
        "http://localhost:8080",

    "llm_model":
        "qwen2.5-0.5b-q4_k_m",

    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    "voice_ref":
        "./test.safetensors",

    "language":
        "french_24l",

    # --------------------------------------------------------
    # Micro
    # --------------------------------------------------------

    "duration":
        5,

    # --------------------------------------------------------
    # Streaming TTS
    # --------------------------------------------------------

    # Nombre de textes en attente maximum
    "tts_queue_size":
        4,

    # Taille minimale avant d'envoyer un morceau au TTS
    "tts_min_chars":
        70,

    # Taille maximale d'un morceau
    "tts_max_chars":
        200,

    # --------------------------------------------------------
    # Queue audio
    # --------------------------------------------------------

    "audio_queue_size":
        300,

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    "n_predict":
        80,

    "temperature":
        0.3,

    "top_p":
        0.85,

    "repeat_penalty":
        1.2,

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    "tts_diagnostic":
        True,

    "llm_diagnostic":
        True,

    "audio_diagnostic":
        True,
}


# ============================================================
# NETTOYAGE TEXTE
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "<|im_start|>",
        ""
    )

    text = text.replace(
        "<|im_end|>",
        ""
    )

    text = text.replace(
        "<|endoftext|>",
        ""
    )

    text = re.sub(
        r'^\s*(assistant|Assistant)\s*:?\s*',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ============================================================
# TEXT CHUNKER
# ============================================================

class TextChunker:

    """
    Transforme les tokens du LLM en morceaux adaptés au TTS.

    IMPORTANT :

    On privilégie les phrases complètes.

    On ne coupe PAS sur les virgules.
    """

    def __init__(
        self,
        min_chars=70,
        max_chars=200
    ):

        self.buffer = ""

        self.min_chars = min_chars
        self.max_chars = max_chars

    def add(self, text):

        results = []

        if not text:
            return results

        self.buffer += text

        while True:

            # ------------------------------------------------
            # Chercher une vraie fin de phrase
            # ------------------------------------------------

            match = re.search(
                r'[.!?…]+(?:["»”\']+)?(?=\s|$)',
                self.buffer
            )

            if match:

                end = match.end()

                candidate = (
                    self.buffer[:end]
                    .strip()
                )

                # ------------------------------------------------
                # Si la phrase est suffisamment longue
                # ------------------------------------------------

                if len(candidate) >= self.min_chars:

                    self.buffer = (
                        self.buffer[end:]
                        .lstrip()
                    )

                    candidate = clean_text(
                        candidate
                    )

                    if candidate:

                        results.append(
                            candidate
                        )

                    continue

            # ------------------------------------------------
            # Buffer trop long
            # ------------------------------------------------

            if len(self.buffer) >= self.max_chars:

                cut = self.buffer.rfind(
                    " ",
                    0,
                    self.max_chars
                )

                if cut > self.min_chars:

                    candidate = (
                        self.buffer[:cut]
                        .strip()
                    )

                    self.buffer = (
                        self.buffer[cut:]
                        .lstrip()
                    )

                    candidate = clean_text(
                        candidate
                    )

                    if candidate:

                        results.append(
                            candidate
                        )

                    continue

            break

        return results

    def flush(self):

        result = clean_text(
            self.buffer
        )

        self.buffer = ""

        return result


# ============================================================
# POCKET-TTS
# ============================================================

class PocketTTS:

    def __init__(
        self,
        voice_ref=None,
        language="french_24l"
    ):

        self.voice_ref = voice_ref
        self.language = language

        self.model = None
        self.voice_state = None

        print()
        print("=" * 60)
        print("🔧 INITIALISATION POCKET-TTS")
        print("=" * 60)

        self._init_tts()

    # --------------------------------------------------------
    # Initialisation
    # --------------------------------------------------------

    def _init_tts(self):

        try:

            print(
                "📦 Chargement du modèle..."
            )

            self.model = (
                TTSModel.load_model(
                    language=self.language
                )
            )

            print(
                "✅ Pocket-TTS chargé"
            )

            print(
                f"   Sample rate : "
                f"{self.model.sample_rate} Hz"
            )

            # ------------------------------------------------
            # Voice
            # ------------------------------------------------

            if (
                self.voice_ref
                and os.path.exists(
                    self.voice_ref
                )
            ):

                print(
                    f"🎤 Voix clonée : "
                    f"{self.voice_ref}"
                )

                self.voice_state = (
                    self.model
                    .get_state_for_audio_prompt(
                        self.voice_ref
                    )
                )

                print(
                    "✅ Voix clonée prête"
                )

            else:

                print(
                    "ℹ️ Voix personnalisée absente"
                )

                print(
                    "   Utilisation de la voix alba"
                )

                self.voice_state = (
                    self.model
                    .get_state_for_audio_prompt(
                        "alba"
                    )
                )

                print(
                    "✅ Voix alba prête"
                )

            # ------------------------------------------------
            # Vérification API streaming
            # ------------------------------------------------

            if hasattr(
                self.model,
                "generate_audio_stream"
            ):

                print(
                    "🚀 generate_audio_stream() disponible"
                )

            else:

                print(
                    "⚠️ generate_audio_stream() "
                    "non disponible"
                )

                print(
                    "   Fallback generate_audio()"
                )

        except Exception as e:

            print(
                f"❌ Erreur Pocket-TTS : {e}"
            )

            import traceback

            traceback.print_exc()

            self.model = None
            self.voice_state = None

    # --------------------------------------------------------
    # Tensor → numpy
    # --------------------------------------------------------

    @staticmethod
    def to_numpy(audio):

        if hasattr(
            audio,
            "detach"
        ):

            audio = (
                audio
                .detach()
                .cpu()
                .numpy()
            )

        elif hasattr(
            audio,
            "numpy"
        ):

            audio = audio.numpy()

        else:

            audio = np.asarray(
                audio
            )

        return np.squeeze(
            audio
        )

    # --------------------------------------------------------
    # Float → PCM16
    # --------------------------------------------------------

    @staticmethod
    def to_pcm16(audio):

        audio = np.asarray(
            audio
        )

        audio = np.nan_to_num(
            audio
        )

        audio = np.clip(
            audio,
            -1.0,
            1.0
        )

        return (
            audio * 32767
        ).astype(
            np.int16
        )

    # --------------------------------------------------------
    # Génération audio streaming
    # --------------------------------------------------------

    def generate_audio_chunks(
        self,
        text
    ):

        if (
            not self.model
            or self.voice_state is None
        ):

            return

        text = clean_text(
            text
        )

        if len(text) < 2:
            return

        print()
        print(
            "🔬 TTS DIAGNOSTIC"
        )

        print(
            f"   Texte : {text}"
        )

        print(
            f"   Longueur : {len(text)} caractères"
        )

        # ----------------------------------------------------
        # Chronométrage
        # ----------------------------------------------------

        t_start = time.perf_counter()

        first_chunk_time = None

        total_samples = 0

        chunk_count = 0

        # ----------------------------------------------------
        # STREAMING
        # ----------------------------------------------------

        if hasattr(
            self.model,
            "generate_audio_stream"
        ):

            try:

                for audio_chunk in (
                    self.model
                    .generate_audio_stream(
                        self.voice_state,
                        text
                    )
                ):

                    now = time.perf_counter()

                    # ----------------------------------------
                    # Premier chunk
                    # ----------------------------------------

                    if first_chunk_time is None:

                        first_chunk_time = now

                        print(
                            f"⚡ Premier chunk audio : "
                            f"{first_chunk_time - t_start:.3f}s",
                            flush=True
                        )

                    audio = self.to_numpy(
                        audio_chunk
                    )

                    if audio.size == 0:
                        continue

                    chunk_count += 1

                    total_samples += (
                        audio.size
                    )

                    duration_audio = (
                        total_samples
                        / self.model.sample_rate
                    )

                    elapsed = (
                        now - t_start
                    )

                    # ----------------------------------------
                    # Diagnostic périodique
                    # ----------------------------------------

                    if (
                        CONFIG[
                            "tts_diagnostic"
                        ]
                        and chunk_count % 5 == 0
                    ):

                        ratio = (
                            duration_audio
                            / elapsed
                            if elapsed > 0
                            else 0
                        )

                        print(
                            f"\r🎵 "
                            f"chunks={chunk_count} | "
                            f"audio={duration_audio:.2f}s | "
                            f"calcul={elapsed:.2f}s | "
                            f"vitesse={ratio:.2f}x",
                            end="",
                            flush=True
                        )

                    yield audio

                # --------------------------------------------
                # Diagnostic final
                # --------------------------------------------

                total_time = (
                    time.perf_counter()
                    - t_start
                )

                audio_duration = (
                    total_samples
                    / self.model.sample_rate
                )

                if total_time > 0:

                    realtime_factor = (
                        audio_duration
                        / total_time
                    )

                else:

                    realtime_factor = 0

                print()

                print(
                    "📊 ===== DIAGNOSTIC TTS ====="
                )

                print(
                    f"   Texte             : "
                    f"{len(text)} caractères"
                )

                print(
                    f"   Chunks audio      : "
                    f"{chunk_count}"
                )

                print(
                    f"   Audio produit     : "
                    f"{audio_duration:.2f}s"
                )

                print(
                    f"   Temps calcul      : "
                    f"{total_time:.2f}s"
                )

                if first_chunk_time:

                    print(
                        f"   Premier audio     : "
                        f"{first_chunk_time - t_start:.3f}s"
                    )

                print(
                    f"   Vitesse TTS       : "
                    f"{realtime_factor:.2f}x"
                )

                if realtime_factor >= 1.5:

                    print(
                        "   ✅ TTS assez rapide"
                    )

                elif realtime_factor >= 1.0:

                    print(
                        "   ⚠️ TTS juste assez rapide"
                    )

                else:

                    print(
                        "   ❌ TTS PLUS LENT "
                        "QUE LA LECTURE"
                    )

                print(
                    "=============================="
                )

                return

            except Exception as e:

                print(
                    f"\n⚠️ Erreur streaming TTS : "
                    f"{e}"
                )

                print(
                    "   Fallback generate_audio()..."
                )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        try:

            t_start = time.perf_counter()

            audio = (
                self.model
                .generate_audio(
                    self.voice_state,
                    text
                )
            )

            audio = self.to_numpy(
                audio
            )

            elapsed = (
                time.perf_counter()
                - t_start
            )

            duration_audio = (
                len(audio)
                / self.model.sample_rate
            )

            ratio = (
                duration_audio
                / elapsed
                if elapsed > 0
                else 0
            )

            print()
            print(
                "📊 ===== DIAGNOSTIC TTS ====="
            )

            print(
                f"   Audio produit : "
                f"{duration_audio:.2f}s"
            )

            print(
                f"   Temps calcul  : "
                f"{elapsed:.2f}s"
            )

            print(
                f"   Vitesse TTS   : "
                f"{ratio:.2f}x"
            )

            print(
                "=============================="
            )

            if audio.size:

                yield audio

        except Exception as e:

            print(
                f"\n❌ Erreur génération TTS : "
                f"{e}"
            )

    # --------------------------------------------------------
    # Start aplay
    # --------------------------------------------------------

    def start_player(self):

        try:

            player = subprocess.Popen(
                [
                    "aplay",

                    "-q",

                    "-t",
                    "raw",

                    "-f",
                    "S16_LE",

                    "-r",
                    str(
                        self.model.sample_rate
                    ),

                    "-c",
                    "1"
                ],

                stdin=subprocess.PIPE,

                stdout=subprocess.DEVNULL,

                stderr=subprocess.DEVNULL
            )

            return player

        except FileNotFoundError:

            print(
                "❌ aplay introuvable."
            )

            print(
                "💡 sudo apt install alsa-utils"
            )

            return None

        except Exception as e:

            print(
                f"❌ Erreur aplay : {e}"
            )

            return None

    # --------------------------------------------------------
    # Stop player
    # --------------------------------------------------------

    def stop_player(
        self,
        player
    ):

        if not player:
            return

        try:

            if player.stdin:
                player.stdin.close()

        except Exception:
            pass

        try:

            player.wait(
                timeout=5
            )

        except Exception:

            try:
                player.kill()
            except Exception:
                pass


# ============================================================
# WHISPER
# ============================================================

def listen():

    duration = CONFIG[
        "duration"
    ]

    print()
    print(
        f"🎤 Parlez pendant "
        f"{duration} secondes..."
    )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp:

            temp_path = tmp.name

        # ----------------------------------------------------
        # Enregistrement
        # ----------------------------------------------------

        subprocess.run(
            [
                "arecord",

                "-d",
                str(duration),

                "-f",
                "S16_LE",

                "-r",
                "16000",

                "-c",
                "1",

                "-t",
                "wav",

                temp_path
            ],

            check=True,

            capture_output=True,

            text=True
        )

        if (
            not os.path.exists(
                temp_path
            )
            or os.path.getsize(
                temp_path
            ) < 1000
        ):

            print(
                "❌ Aucun audio enregistré"
            )

            return ""

        # ----------------------------------------------------
        # Vérification niveau
        # ----------------------------------------------------

        try:

            import wave

            with wave.open(
                temp_path,
                "rb"
            ) as wav:

                frames = (
                    wav.readframes(
                        wav.getnframes()
                    )
                )

            audio_array = np.frombuffer(
                frames,
                dtype=np.int16
            )

            if audio_array.size:

                max_amplitude = int(
                    np.max(
                        np.abs(
                            audio_array
                        )
                    )
                )

                print(
                    f"📊 Amplitude max : "
                    f"{max_amplitude}"
                )

                if max_amplitude < 100:

                    print(
                        "❌ Signal microphone "
                        "trop faible"
                    )

                    return ""

        except Exception as e:

            print(
                f"⚠️ Analyse audio : {e}"
            )

        print(
            "🎤 Audio enregistré"
        )

        print(
            "🧠 Transcription..."
        )

        # ----------------------------------------------------
        # Whisper
        # ----------------------------------------------------

        result = subprocess.run(
            [
                CONFIG[
                    "whisper_cli"
                ],

                "-m",
                CONFIG[
                    "whisper_model"
                ],

                "-f",
                temp_path,

                "-l",
                "fr",

                "--no-timestamps",

                "--print-progress"
            ],

            capture_output=True,

            text=True
        )

        text = result.stdout.strip()

        text = re.sub(
            r'\[[^\]]+\]',
            ' ',
            text
        )

        text = re.sub(
            r'\s+',
            ' ',
            text
        )

        text = text.strip()

        if text:

            print(
                f"📝 Vous : {text}"
            )

            return text

        print(
            "❌ Aucune parole détectée"
        )

        return ""

    except subprocess.CalledProcessError as e:

        print(
            f"❌ Erreur arecord/Whisper : "
            f"{e}"
        )

        return ""

    except Exception as e:

        print(
            f"❌ Erreur écoute : {e}"
        )

        return ""

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            try:

                os.unlink(
                    temp_path
                )

            except Exception:
                pass


# ============================================================
# LLM STREAMING
# ============================================================

def think_stream(question):

    if not question:
        return

    print()
    print(
        "🧠 LLM : ",
        end="",
        flush=True
    )

    # --------------------------------------------------------
    # Chronomètre LLM
    # --------------------------------------------------------

    llm_start = time.perf_counter()

    first_token_time = None

    token_count = 0

    full_response = ""

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    messages = [

        {
            "role":
                "system",

            "content":
                (
                    "Tu es un assistant vocal "
                    "utile et naturel. "
                    "Réponds en français. "
                    "Réponds avec une ou deux "
                    "phrases courtes. "
                    "Pas de markdown. "
                    "Pas de listes. "
                    "Ne répète pas la question."
                )
        },

        {
            "role":
                "user",

            "content":
                question
        }
    ]

    payload = {

        "model":
            CONFIG[
                "llm_model"
            ],

        "messages":
            messages,

        "stream":
            True,

        "max_tokens":
            CONFIG[
                "n_predict"
            ],

        "temperature":
            CONFIG[
                "temperature"
            ],

        "top_p":
            CONFIG[
                "top_p"
            ],

        "repeat_penalty":
            CONFIG[
                "repeat_penalty"
            ]
    }

    try:

        response = requests.post(
            (
                f"{CONFIG['llm_url']}"
                "/v1/chat/completions"
            ),

            json=payload,

            stream=True,

            timeout=60,

            headers={
                "Accept":
                    "text/event-stream"
            }
        )

        print()
        print(
            f"🔌 HTTP LLM : "
            f"{response.status_code}"
        )

        if response.status_code != 200:

            print(
                response.text[:2000]
            )

            return

        # ----------------------------------------------------
        # Chunker
        # ----------------------------------------------------

        chunker = TextChunker(
            min_chars=CONFIG[
                "tts_min_chars"
            ],

            max_chars=CONFIG[
                "tts_max_chars"
            ]
        )

        # ----------------------------------------------------
        # SSE
        # ----------------------------------------------------

        for line in response.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue

            if isinstance(
                line,
                bytes
            ):

                line = line.decode(
                    "utf-8",
                    errors="replace"
                )

            line = line.strip()

            if line.startswith(":"):
                continue

            if line.startswith(
                "data:"
            ):

                data = (
                    line[5:]
                    .strip()
                )

            else:

                data = line

            if data == "[DONE]":

                break

            if not data:

                continue

            try:

                obj = json.loads(
                    data
                )

            except json.JSONDecodeError:

                continue

            choices = obj.get(
                "choices",
                []
            )

            if not choices:
                continue

            choice = choices[0]

            # ------------------------------------------------
            # OpenAI compatible format
            # ------------------------------------------------

            delta = choice.get(
                "delta",
                {}
            )

            token = delta.get(
                "content",
                ""
            )

            # ------------------------------------------------
            # Fallback llama.cpp
            # ------------------------------------------------

            if not token:

                token = choice.get(
                    "text",
                    ""
                )

            if not token:
                continue

            # ------------------------------------------------
            # Premier token
            # ------------------------------------------------

            token_count += 1

            if first_token_time is None:

                first_token_time = (
                    time.perf_counter()
                )

                if CONFIG[
                    "llm_diagnostic"
                ]:

                    print()
                    print(
                        f"⚡ Premier token LLM : "
                        f"{first_token_time - llm_start:.3f}s"
                    )

            # ------------------------------------------------
            # Texte complet
            # ------------------------------------------------

            full_response += token

            # ------------------------------------------------
            # Affichage
            # ------------------------------------------------

            print(
                token,
                end="",
                flush=True
            )

            # ------------------------------------------------
            # Chunk TTS
            # ------------------------------------------------

            chunks = chunker.add(
                token
            )

            for chunk in chunks:

                yield (
                    chunk,
                    full_response
                )

        # ----------------------------------------------------
        # Dernier morceau
        # ----------------------------------------------------

        remaining = chunker.flush()

        if remaining:

            yield (
                remaining,
                full_response
            )

        # ----------------------------------------------------
        # Diagnostic LLM
        # ----------------------------------------------------

        llm_total = (
            time.perf_counter()
            - llm_start
        )

        print()

        if CONFIG[
            "llm_diagnostic"
        ]:

            print(
                "📊 ===== DIAGNOSTIC LLM ====="
            )

            print(
                f"   Tokens : "
                f"{token_count}"
            )

            print(
                f"   Temps  : "
                f"{llm_total:.2f}s"
            )

            if (
                token_count > 0
                and llm_total > 0
            ):

                print(
                    f"   Vitesse : "
                    f"{token_count / llm_total:.1f} "
                    f"tokens/s"
                )

            if first_token_time:

                print(
                    f"   Premier token : "
                    f"{first_token_time - llm_start:.3f}s"
                )

            print(
                "=============================="
            )

    except requests.RequestException as e:

        print(
            f"\n❌ Erreur HTTP LLM : {e}"
        )

    except Exception as e:

        print(
            f"\n❌ Erreur LLM : {e}"
        )

        import traceback

        traceback.print_exc()


# ============================================================
# AUDIO PLAYER THREAD
# ============================================================

def audio_player_worker(
    tts,
    audio_queue,
    stop_event
):

    """
    Lit les chunks audio sans attendre la génération
    complète du TTS.
    """

    player = tts.start_player()

    if player is None:

        stop_event.set()

        return

    chunk_counter = 0

    try:

        while True:

            try:

                item = audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                if stop_event.is_set():

                    break

                continue

            # ------------------------------------------------
            # Fin
            # ------------------------------------------------

            if item is None:

                audio_queue.task_done()

                break

            chunk_counter += 1

            # ------------------------------------------------
            # Diagnostic queue
            # ------------------------------------------------

            if (
                CONFIG[
                    "audio_diagnostic"
                ]
                and chunk_counter % 20 == 0
            ):

                print(
                    f"\n🎧 Player | "
                    f"queue={audio_queue.qsize()} "
                    f"chunks",
                    flush=True
                )

            # ------------------------------------------------
            # Lecture
            # ------------------------------------------------

            try:

                player.stdin.write(
                    item.tobytes()
                )

                player.stdin.flush()

            except BrokenPipeError:

                print(
                    "\n⚠️ Pipe audio fermé"
                )

                stop_event.set()

                break

            finally:

                audio_queue.task_done()

    finally:

        tts.stop_player(
            player
        )


# ============================================================
# TTS WORKER
# ============================================================

def tts_worker(
    tts_queue,
    audio_queue,
    tts,
    stop_event
):

    """
    Génère les morceaux TTS dans un thread séparé.

    Le player audio peut donc lire pendant que Pocket-TTS
    génère le morceau suivant.
    """

    try:

        while True:

            text = tts_queue.get()

            # ------------------------------------------------
            # Fin
            # ------------------------------------------------

            if text is None:

                tts_queue.task_done()

                break

            try:

                print()
                print(
                    f"🎙️ Synthèse : {text}",
                    flush=True
                )

                # --------------------------------------------
                # Génération streaming
                # --------------------------------------------

                for audio in (
                    tts.generate_audio_chunks(
                        text
                    )
                ):

                    if stop_event.is_set():

                        break

                    pcm = (
                        tts.to_pcm16(
                            audio
                        )
                    )

                    if pcm.size == 0:
                        continue

                    # ----------------------------------------
                    # IMPORTANT
                    #
                    # Aucun silence artificiel ici.
                    #
                    # Les chunks internes de Pocket-TTS
                    # doivent rester collés.
                    # ----------------------------------------

                    audio_queue.put(
                        pcm
                    )

            except Exception as e:

                print(
                    f"\n❌ Erreur TTS : "
                    f"{e}"
                )

            finally:

                tts_queue.task_done()

    finally:

        # ----------------------------------------------------
        # Signaler au player qu'il n'y aura plus d'audio
        # ----------------------------------------------------

        audio_queue.put(
            None
        )


# ============================================================
# ASSISTANT VOCAL
# ============================================================

class VocalAssistant:

    def __init__(self):

        print()
        print("=" * 60)
        print(
            "🤖 ASSISTANT VOCAL STREAMING"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        if self.check_llm():

            print(
                "✅ Serveur llama.cpp disponible"
            )

        else:

            print(
                "⚠️ Serveur LLM non disponible"
            )

            print()
            print(
                "Lancez dans un autre terminal :"
            )

            print()

            print(
                "cd ./llama.cpp && "
                "./build/bin/llama-server "
                "-m ./models/"
                "qwen2.5-0.5b-q4_k_m.gguf "
                "-c 2048 "
                "--host 0.0.0.0 "
                "--port 8080"
            )

            input(
                "\nAppuyez sur Entrée "
                "quand le serveur est prêt..."
            )

        # ----------------------------------------------------
        # Voice
        # ----------------------------------------------------

        if not os.path.exists(
            CONFIG[
                "voice_ref"
            ]
        ):

            print(
                f"⚠️ Voix absente : "
                f"{CONFIG['voice_ref']}"
            )

            print(
                "   Utilisation de alba."
            )

            CONFIG[
                "voice_ref"
            ] = None

        # ----------------------------------------------------
        # TTS
        # ----------------------------------------------------

        self.tts = PocketTTS(
            voice_ref=CONFIG[
                "voice_ref"
            ],

            language=CONFIG[
                "language"
            ]
        )

        if (
            self.tts.model
            and self.tts.voice_state
        ):

            print()
            print(
                "✅ Synthèse vocale prête"
            )

        else:

            print()
            print(
                "❌ Synthèse vocale indisponible"
            )

        print()
        print("=" * 60)
        print(
            "🔊 ASSISTANT PRÊT"
        )
        print(
            "   Ctrl+C pour arrêter"
        )
        print("=" * 60)

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    def check_llm(self):

        try:

            response = requests.get(
                (
                    f"{CONFIG['llm_url']}"
                    "/health"
                ),

                timeout=2
            )

            return (
                response.status_code == 200
            )

        except Exception:

            return False

    # --------------------------------------------------------
    # Une interaction
    # --------------------------------------------------------

    def run_once(self):

        # ====================================================
        # 1. Écoute
        # ====================================================

        question = listen()

        if not question:

            print(
                "-" * 60
            )

            return

        # ====================================================
        # 2. Queues
        # ====================================================

        tts_queue = queue.Queue(
            maxsize=CONFIG[
                "tts_queue_size"
            ]
        )

        audio_queue = queue.Queue(
            maxsize=CONFIG[
                "audio_queue_size"
            ]
        )

        stop_event = threading.Event()

        # ====================================================
        # 3. AUDIO PLAYER
        # ====================================================

        player_thread = threading.Thread(
            target=audio_player_worker,

            args=(
                self.tts,
                audio_queue,
                stop_event
            ),

            daemon=True
        )

        player_thread.start()

        # ====================================================
        # 4. TTS WORKER
        # ====================================================

        tts_thread = threading.Thread(
            target=tts_worker,

            args=(
                tts_queue,
                audio_queue,
                self.tts,
                stop_event
            ),

            daemon=True
        )

        tts_thread.start()

        # ====================================================
        # 5. LLM
        # ====================================================

        full_response = ""

        try:

            for chunk, current_response in (
                think_stream(
                    question
                )
            ):

                full_response = (
                    current_response
                )

                if chunk:

                    # ----------------------------------------
                    # Envoyer au TTS
                    # ----------------------------------------

                    tts_queue.put(
                        chunk
                    )

        except KeyboardInterrupt:

            print(
                "\n🛑 Arrêt..."
            )

            stop_event.set()

        except Exception as e:

            print(
                f"\n❌ Erreur interaction : "
                f"{e}"
            )

            stop_event.set()

        finally:

            # =================================================
            # Fin LLM
            # =================================================

            tts_queue.put(
                None
            )

            # =================================================
            # Attendre la fin TTS
            # =================================================

            tts_thread.join()

            # =================================================
            # Attendre la fin audio
            # =================================================

            player_thread.join()

        # ====================================================
        # 6. Réponse finale
        # ====================================================

        full_response = clean_text(
            full_response
        )

        print()

        if full_response:

            print(
                f"🤖 Réponse : "
                f"{full_response}"
            )

        else:

            print(
                "❌ Le LLM n'a produit "
                "aucun texte."
            )

        print(
            "-" * 60
        )

    # --------------------------------------------------------
    # Boucle principale
    # --------------------------------------------------------

    def run_loop(self):

        try:

            while True:

                self.run_once()

                time.sleep(
                    0.2
                )

        except KeyboardInterrupt:

            print()
            print(
                "👋 Au revoir !"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Assistant vocal "
            "LLM + Whisper + Pocket-TTS"
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Une seule interaction"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Durée d'enregistrement"
    )

    parser.add_argument(
        "--voice",
        type=str,
        help="Fichier de référence vocal"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    CONFIG[
        "duration"
    ] = args.duration

    if args.voice:

        CONFIG[
            "voice_ref"
        ] = args.voice

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    assistant = VocalAssistant()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    if args.once:

        assistant.run_once()

    else:

        assistant.run_loop()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()