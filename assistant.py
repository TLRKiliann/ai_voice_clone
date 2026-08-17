#!/usr/bin/env python3
"""
Assistant Vocal - Version avec clonage vocal via pocket-tts (API CORRECTE)
Utilise get_state_for_audio_prompt() et generate_audio()
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
import scipy.io.wavfile
from pocket_tts import TTSModel

# qwen2.5-1.5b-instruct-q4_k_m.gguf -c 2048
CONFIG = {
    "whisper_cli": "./whisper.cpp/build/bin/whisper-cli",
    "whisper_model": "./whisper.cpp/models/ggml-base.bin",
    "whisper_threads": 4,
    "llm_url": "http://localhost:8080",
    "llm_server_cmd": "cd ./llama.cpp && ./build/bin/llama-server -m ../models/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 2048 --host 0.0.0.0 --port 8080",
    "voice_ref": "./test.safetensors",
    "duration": 5,
    "language": "french_24l",
    "quantize": True
}

def clean_llm_response(text):
    """Nettoie et formate la réponse du LLM pour la synthèse vocale."""
    
    if not text:
        return "Je n'ai pas de réponse."
    
    # 1. Supprimer les tokens spéciaux du format Qwen
    text = re.sub(r'<\|im_[a-z]+\|>', '', text)
    text = re.sub(r'\[INST\]|\[/INST\]', '', text)
    
    # 2. Supprimer les marqueurs de rôle (user/assistant)
    text = re.sub(r'\b(user|assistant|Assistant|system|System)\b\s*[:]?\s*', '', text, flags=re.IGNORECASE)
    
    # 3. Nettoyer les artefacts de ponctuation
    text = re.sub(r'\s+', ' ', text)  # Multi-espaces → un seul
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Espace avant ponctuation
    
    # 4. Supprimer les tirets de début/fin de ligne
    text = re.sub(r'^[\s-]+|[\s-]+$', '', text)
    
    # 5. Découper en phrases (gère ., !, ?)
    sentence_endings = r'[.!?]'
    sentences = []
    for part in re.split(sentence_endings, text):
        part = part.strip()
        if part and len(part) > 2 and not part.startswith(('http', 'www')):
            sentences.append(part)
    
    if not sentences:
        return "Je n'ai pas de réponse."
    
    # 6. Prendre les 2 premières phrases
    text = ". ".join(sentences[:2])
    
    # 7. Ajouter la ponctuation finale
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    # 8. Limiter à 150 caractères (ou configurable)
    max_length = 150
    if len(text) > max_length:
        # Couper au dernier espace ou ponctuation
        cut_pos = text.rfind(' ', 0, max_length)
        if cut_pos == -1:
            cut_pos = max_length
        text = text[:cut_pos] + "..."
    
    return text

# ============ POCKET-TTS AVEC CLONAGE ============
class PocketTTS:
    def __init__(self, voice_ref=None, language="french_24l", quantize=True):
        """Initialise Pocket-TTS avec clonage vocal"""
        self.voice_ref = voice_ref
        self.language = language
        self.quantize = quantize
        self.model = None
        self.voice_state = None
        
        print("🔧 Initialisation de Pocket-TTS...")
        print(f"🔧 Configuration TTS: language={self.language}, quantize={self.quantize}")

        self._init_tts()
    
    def _init_tts(self):
        """Initialise le moteur TTS avec les bonnes méthodes"""
        try:
            print("📦 Chargement du modèle TTS avec load_model()...")
            self.model = TTSModel.load_model(
                language=self.language,
                quantize=self.quantize
            )
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

            time.sleep(0.1)

            file_size = os.path.getsize(temp_path)
            print(f"   Taille du fichier: {file_size} octets")
            
            if file_size > 1000:
                print("🔊 Lecture de l'audio...")
                try:
                    subprocess.run(["aplay", temp_path], check=True)
                    print("✅ Audio joué avec succès")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ Erreur lors de la lecture audio: {e}")
                    print(f"   Fichier sauvegardé: {temp_path}")
                    return False
            else:
                print("❌ Fichier audio trop petit")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de la synthèse: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print("🧹 Fichier temporaire supprimé")
            except:
                pass

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
            "-f", "S16_LE",         # Format 16-bit
            "-r", "16000",          # 16kHz (recommandé pour Whisper)
            "-c", "1",              # Mono
            "-t", "wav",
            temp_path               # Spécifier le fichier de sortie
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

        result = subprocess.run([
            CONFIG["whisper_cli"],
            "-m", CONFIG["whisper_model"],
            "-f", temp_path,
            "-l", "fr",
            "--no-timestamps",
            "--no-gpu",
            "--threads", str(CONFIG["whisper_threads"]),
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
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        response = requests.post(
            f"{CONFIG['llm_url']}/completion",
            headers=headers,
            json={
                "prompt": full_prompt,
                "n_predict": 80, # 50 standard
                "temperature": 0.5,
                "top_p": 0.85,
                "min_p": 0.05,
                "repeat_penalty": 1.1,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.2,
                "stop": ["<|im_end|>", "\nuser", "\n\n"],
                "stream": False,
                "cache_prompt": True,
                "ignore_eos": False,
                "logit_bias": []  # Pour exclure certains tokens si nécessaire
            },
            timeout=(5.0, 30.0)  # Connexion + lecture
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
        self.tts = PocketTTS(
            voice_ref=CONFIG["voice_ref"],
            language=CONFIG["language"],
            quantize=CONFIG["quantize"]
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
        question = listen()
        if question:
            reponse = think(question)
            if reponse:
                print(f"\n🤖 Assistant: {reponse}")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--voice", type=str, help="Fichier de référence pour le clonage")
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