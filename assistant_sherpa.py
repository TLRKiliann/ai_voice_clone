#!/usr/bin/env python3
"""
Assistant Vocal - Version avec streaming (sherpa-onnx) + clonage vocal (pocket-tts)
"""

import os
import subprocess
import requests
import tempfile
import time
import argparse
import soundfile as sf
import re
import numpy as np
import pyaudio  # Pour le streaming
# import wave
from pocket_tts import TTSModel

# ============ CONFIGURATION ============
CONFIG = {
    "whisper_cli": "./whisper.cpp/build/bin/whisper-cli",
    "whisper_model": "./whisper.cpp/models/ggml-base.bin",
    "llm_url": "http://localhost:8080",
    "llm_server_cmd": "cd ./llama.cpp && ./build/bin/llama-server -m ./models/qwen2.5-1.5b-q4_k_m.gguf -c 2048 --host 0.0.0.0 --port 8080",
    "voice_ref": "./test.safetensors",
    "duration": 5,
    "language": "french_24l",
    "sample_rate": 16000,
    "chunk_duration": 0.2,
    "silence_timeout": 2.0,
}

def clean_llm_response(text):
    if not text:
        return "Je n'ai pas de réponse."
    text = re.sub(r'-[a-zA-Z]+\s+', ' ', text)
    text = re.sub(r'-[a-zA-Z]+$', '', text)
    text = re.sub(r'(user|assistant|Assistant|\[\])', '', text)
    sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 2]
    if not sentences:
        return "Je n'ai pas de réponse."
    text = ". ".join(sentences[:2])
    if not text.endswith('.'):
        text += '.'
    if len(text) > 150:
        text = text[:150] + "..."
    return text

# ============ POCKET-TTS AVEC CLONAGE (API CORRECTE) ============
class PocketTTS:
    def __init__(self, voice_ref=None, language="french_24l"):
        """Initialise Pocket-TTS avec clonage vocal"""
        self.voice_ref = voice_ref
        self.language = language
        self.model = None
        self.voice_state = None
        
        print("🔧 Initialisation de Pocket-TTS...")
        self._init_tts()
    
    def _init_tts(self):
        """Initialise le moteur TTS avec les bonnes méthodes"""
        try:
            print("📦 Chargement du modèle TTS avec load_model()...")
            self.model = TTSModel.load_model(language=self.language)
            print("✅ Pocket-TTS initialisé avec succès")
            
            if self.model:
                print(f"   Sample rate du modèle: {self.model.sample_rate}Hz")
                
                # Charger la voix de référence (ou une voix par défaut)
                if self.voice_ref and os.path.exists(self.voice_ref):
                    print(f"🎤 Chargement de la voix de référence: {self.voice_ref}")
                    self.voice_state = self.model.get_state_for_audio_prompt(self.voice_ref)
                    print("✅ Voix clonée chargée avec succès")
                else:
                    print("ℹ️ Aucune voix de référence trouvée, utilisation de la voix 'alba'")
                    self.voice_state = self.model.get_state_for_audio_prompt("alba")
                    print("✅ Voix par défaut chargée")
                    
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.voice_state = None
    
    def speak(self, text):
        """Génère l'audio avec clonage vocal"""
        if not self.model or self.voice_state is None:
            print("❌ TTS non initialisé ou voix non chargée")
            return False
        
        if not text:
            print("❌ Texte vide")
            return False
        
        text = clean_llm_response(text)
        if len(text) < 3:
            print("⚠️ Texte trop court")
            return False
        
        print(f"🗣️ Synthèse: {text[:60]}...")
        
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp.name
        temp.close()
        
        try:
            print("🎵 Génération de l'audio avec generate_audio()...")
            audio_tensor = self.model.generate_audio(self.voice_state, text)
            
            # Convertir en numpy array pour l'export
            if hasattr(audio_tensor, 'numpy'):
                audio_data = audio_tensor.numpy()
            else:
                audio_data = np.array(audio_tensor)
            
            print(f"💾 Sauvegarde de l'audio ({len(audio_data)} échantillons)...")
            sf.write(temp_path, audio_data, self.model.sample_rate)
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(temp_path)
            print(f"   Taille du fichier: {file_size} octets")
            
            if file_size > 1000:
                # Jouer l'audio
                print("🔊 Lecture de l'audio...")
                try:
                    subprocess.run(["aplay", temp_path], check=True)
                    print("✅ Audio joué avec succès")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Erreur lors de la lecture audio: {e}")
                    print(f"   Fichier sauvegardé: {temp_path}")
                    return True
                except FileNotFoundError:
                    print("⚠️ 'aplay' non trouvé, essayez avec 'paplay' ou 'play'")
                    try:
                        subprocess.run(["paplay", temp_path], check=True)
                        print("✅ Audio joué avec paplay")
                        return True
                    except:
                        print(f"   Fichier sauvegardé: {temp_path}")
                        return True
            else:
                print("❌ Fichier audio trop petit")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de la synthèse: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============ NOUVELLE CLASSE POUR LA RECONNAISSANCE STREAMING ============
class StreamingRecognizer:
    """Reconnaissance vocale en streaming avec sherpa-onnx"""
    
    def __init__(self):
        self.recognizer = None
        self.stream = None
        self.pyaudio_instance = None
        self._init_sherpa()
    
    def _get_text_from_result(self, result):
        """Extrait le texte d'un résultat, qu'il soit string ou objet"""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if hasattr(result, 'text'):
            return result.text
        return str(result)
    
    def _init_sherpa(self):
        """Initialise sherpa-onnx avec un modèle streaming français"""
        try:
            import sherpa_onnx
            
            # Chemin vers le dossier du modèle
            model_dir = "./models/sherpa-onnx-streaming-zipformer-fr-2023-04-14"
            
            # Vérifier que les fichiers existent
            token_path = f"{model_dir}/tokens.txt"
            encoder_path = f"{model_dir}/encoder-epoch-29-avg-9-with-averaged-model.int8.onnx"
            decoder_path = f"{model_dir}/decoder-epoch-29-avg-9-with-averaged-model.int8.onnx"
            joiner_path = f"{model_dir}/joiner-epoch-29-avg-9-with-averaged-model.int8.onnx"
            
            if not os.path.exists(token_path):
                raise FileNotFoundError(f"Fichier tokens.txt non trouvé dans {model_dir}")
            
            # Utiliser la méthode from_transducer
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=token_path,
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                num_threads=4,
                sample_rate=CONFIG["sample_rate"],
                decoding_method="greedy_search",
                max_active_paths=4,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=20.0
            )
            
            print("✅ Modèle sherpa-onnx (Zipformer FR) chargé avec succès")
            print("   Fin de phrase détectée après 1.2s ou 2.4s de silence")
            
        except ImportError:
            print("⚠️ sherpa-onnx non installé. Installez: pip install sherpa-onnx")
            self.recognizer = None
        except Exception as e:
            print(f"⚠️ Erreur de chargement du modèle: {e}")
            import traceback
            traceback.print_exc()
            self.recognizer = None
    
    def listen_streaming(self, duration=5):
        """
        Écoute en streaming avec détection automatique de la fin de parole.
        Si sherpa n'est pas disponible, utilise whisper-cli en fallback.
        """
        # FALLBACK : si sherpa n'est pas disponible, utiliser l'ancienne méthode
        if self.recognizer is None:
            print("📣 Utilisation du mode fallback (whisper-cli)")
            return self._listen_fallback(duration)
        
        print("🎤 Parlez... (détection automatique)")
        
        # Initialiser PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=CONFIG["sample_rate"],
            input=True,
            frames_per_buffer=int(CONFIG["sample_rate"] * CONFIG["chunk_duration"])
        )
        
        # Créer un flux de reconnaissance
        online_stream = self.recognizer.create_stream()
        text_acc = []
        silence_counter = 0
        max_silence_chunks = int(CONFIG["silence_timeout"] / CONFIG["chunk_duration"])
        
        try:
            while True:
                # Lire un chunk audio
                data = self.stream.read(
                    int(CONFIG["sample_rate"] * CONFIG["chunk_duration"]),
                    exception_on_overflow=False
                )
                audio_np = np.frombuffer(data, dtype=np.int16)
                
                # Envoyer au reconnaisseur
                online_stream.accept_waveform(CONFIG["sample_rate"], audio_np)
                
                # Décoder
                while self.recognizer.is_ready(online_stream):
                    self.recognizer.decode_stream(online_stream)
                
                # Vérifier la fin de la phrase
                if self.recognizer.is_endpoint(online_stream):
                    result = self.recognizer.get_result(online_stream)
                    text = self._get_text_from_result(result).strip()
                    
                    if text:
                        print(f"📝 Vous: {text}")
                        return text
                    
                    # Réinitialiser
                    online_stream = self.recognizer.create_stream()
                    silence_counter = 0
                
                # Détection de silence prolongé (fallback)
                is_silent = np.max(np.abs(audio_np)) < 100
                if is_silent:
                    silence_counter += 1
                else:
                    silence_counter = 0
                    
                    # Accumuler le texte partiel (optionnel)
                    result = self.recognizer.get_result(online_stream)
                    partial = self._get_text_from_result(result).strip()
                    if partial and partial not in text_acc:
                        text_acc.append(partial)
                        # Afficher la progression
                        print(f"\r📝 ... {partial}", end="")
                
                # Si trop de silence, on considère que c'est fini
                if silence_counter > max_silence_chunks and len(text_acc) > 0:
                    final_text = " ".join(text_acc).strip()
                    if final_text:
                        print(f"\n📝 Vous: {final_text}")
                        return final_text
                
        except KeyboardInterrupt:
            return ""
        except Exception as e:
            print(f"❌ Erreur streaming: {e}")
            import traceback
            traceback.print_exc()
            return ""
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
    
    def _listen_fallback(self, duration):
        """Ancienne méthode avec arecord + whisper-cli"""
        print(f"\n🎤 Parlez pendant {duration} secondes...")
        
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            subprocess.run([
                "arecord", "-d", str(duration), "-f", "S16_LE",
                "-r", str(CONFIG["sample_rate"]), "-c", "1", "-t", "wav", temp_path
            ], check=True, capture_output=True)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return ""
            
            # Transcription avec whisper-cli
            result = subprocess.run([
                CONFIG["whisper_cli"],
                "-m", CONFIG["whisper_model"],
                "-f", temp_path,
                "-l", "fr",
                "--no-timestamps",
                "--print-progress",
            ], capture_output=True, text=True)
            
            text = result.stdout.strip()
            if text:
                print(f"📝 Vous: {text}")
            return text
            
        except Exception as e:
            print(f"❌ Erreur fallback: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

# ============ COMPOSANTS (LISTEN, THINK, ASSISTANT) ============
def listen():
    duration = CONFIG["duration"]
    print(f"\n🎤 Parlez pendant {duration} secondes...")
    
    temp_path = None
    
    try:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name

        # CORRECTION : Rediriger l'audio vers le fichier
        result = subprocess.run([
            "arecord", 
            "-d", str(duration), 
            "-f", "S16_LE",        # Format 16-bit
            "-r", "16000",          # 16kHz (recommandé pour Whisper)
            "-c", "1",              # Mono
            "-t", "wav",
            temp_path              # Spécifier le fichier de sortie
        ], check=True, capture_output=True, text=True)
        
        # Vérifier que le fichier a été créé
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print("❌ Aucun audio enregistré")
            return ""
        
        # Vérifier le niveau audio
        try:
            import wave
            import numpy as np
            
            with wave.open(temp_path, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                if len(frames) < 100:
                    print("❌ Audio trop court")
                    return ""
                
                audio_array = np.frombuffer(frames, dtype=np.int16)
                max_amplitude = np.max(np.abs(audio_array))
                print(f"📊 Amplitude max: {max_amplitude}")
                
                if max_amplitude < 100:
                    print("❌ Signal trop faible (microphone silencieux ou mal configuré)")
                    print("💡 Vérifiez que votre microphone est branché et actif")
                    return ""
        except Exception as e:
            print(f"⚠️ Erreur d'analyse audio: {e}")
        
        print("🎤 Audio enregistré, transcription en cours...")
        
        # Transcrire avec Whisper
        result = subprocess.run([
            CONFIG["whisper_cli"],
            "-m", CONFIG["whisper_model"],
            "-f", temp_path,
            "-l", "fr",
            "--no-timestamps",
            "--print-progress",
        ], capture_output=True, text=True)
        
        text = result.stdout.strip()
        if text:
            print(f"📝 Vous: {text}")
        else:
            print("❌ Aucune parole détectée")
            if result.stderr:
                print(f"Erreur Whisper: {result.stderr}")
        return text
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur d'enregistrement: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return ""
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return ""
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass

# ============ CLASSE VOCALASSISTANT MODIFIÉE ============
def think(prompt):
    if not prompt:
        return "Je n'ai rien compris."
    
    print("🧠 Réflexion...")
    
    full_prompt = f"""<|im_start|>system
Vous êtes un assistant utile. Répondez en une phrase courte et naturelle, sans répéter la question.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
    try:
        response = requests.post(
            f"{CONFIG['llm_url']}/completion",
            json={
                "prompt": full_prompt,
                "n_predict": 50,
                "temperature": 0.3,
                "top_p": 0.85,
                "repeat_penalty": 2.0,
                "stop": ["<|im_end|>", "\nuser", "\n\n"],
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json().get("content", "").strip()
            cleaned = clean_llm_response(content)
            if len(cleaned) < 3:
                return "Je n'ai pas de réponse à cette question."
            return cleaned
        return "Désolé, le serveur a rencontré une erreur."
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return "Je rencontre une difficulté technique."

class VocalAssistant:
    def __init__(self):
        print("\n" + "="*50)
        print("🤖 ASSISTANT VOCAL AVEC CLONAGE")
        print("="*50)
        
        self.llm_ready = self._check_llm()
        if not self.llm_ready:
            print("⚠️ Serveur LLM non démarré.")
            print(f"💡 Lancez-le dans un autre terminal:")
            print(f"   {CONFIG['llm_server_cmd']}")
            input("Appuyez sur Entrée quand prêt...")
        
        if not os.path.exists(CONFIG["voice_ref"]):
            print(f"⚠️ Voix de référence non trouvée: {CONFIG['voice_ref']}")
            print("💡 Créez-en une avec:")
            print("   arecord -d 5 -f cd -r 16000 -c 1 ma_voix.wav")
            CONFIG["voice_ref"] = None
        
        print("\n" + "="*50)
        print("🎤 INITIALISATION DU TTS")
        print("="*50)
        self.recognizer = StreamingRecognizer()
        self.tts = PocketTTS(
            voice_ref=CONFIG["voice_ref"],
            language=CONFIG["language"]
        )
        
        if self.tts.model and self.tts.voice_state is not None:
            print("✅ Synthèse vocale avec clonage prête")
        else:
            print("⚠️ Synthèse vocale désactivée")
        
        print("\n🔊 Assistant prêt ! (Ctrl+C pour arrêter)")
    
    def _check_llm(self):
        try:
            response = requests.get(f"{CONFIG['llm_url']}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def run_once(self):
        """Exécute un cycle complet: écoute -> pense -> répond"""
        
        # 1. ÉCOUTE - Utilise la nouvelle méthode streaming
        question = self.recognizer.listen_streaming(CONFIG["duration"])
        
        if not question:
            print("ℹ️ Aucune parole détectée")
            return
        
        # 2. PENSE
        reponse = think(question)
        if not reponse:
            return
        
        print(f"\n🤖 Assistant: {reponse}")
        # 3. RÉPOND (TTS)
        if self.tts and self.tts.model and self.tts.voice_state is not None:
            self.tts.speak(reponse)
        else:
            print("🔇 Synthèse vocale désactivée")
        
        print("-"*50)
    
    def run_loop(self):
        try:
            while True:
                self.run_once()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")

# ============ FONCTION MAIN ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--voice", type=str, help="Fichier de référence pour le clonage")
    parser.add_argument("--no-streaming", action="store_true", help="Utilise whisper-cli (fallback)")
    args = parser.parse_args()
    
    if args.voice:
        CONFIG["voice_ref"] = args.voice
    CONFIG["duration"] = args.duration
    
    assistant = VocalAssistant()
    
    if args.once:
        assistant.run_once()
    else:
        assistant.run_loop()

if __name__ == "__main__":
    main()
