#!/usr/bin/env python3
"""
Script de démonstration pour le streaming audio avec Pocket-TTS.

Ce script montre comment utiliser generate_audio_stream() pour générer
de l'audio en temps réel avec une latence minimale.
"""

import argparse
import time
import sys
from pathlib import Path
import numpy as np

try:
    import torch
    import scipy.io.wavfile
    from pocket_tts import TTSModel
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    print("📦 Assurez-vous que les paquets suivants sont installés :")
    print("   pip install pocket-tts torch scipy")
    sys.exit(1)

# === CONFIGURATION ===
SAMPLE_RATE = 24000  # 24 kHz, taux d'échantillonnage de Pocket-TTS
CHUNK_DURATION_MS = 80  # ~80 ms par chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000.0))  # ~1920 samples

# === FONCTIONS UTILITAIRES ===

def play_audio_chunk(audio_chunk, sample_rate=SAMPLE_RATE):
    """
    Joue un chunk audio en temps réel (nécessite sounddevice).
    Alternative : utiliser pyaudio.
    """
    try:
        import sounddevice as sd
        sd.play(audio_chunk.numpy(), sample_rate, blocking=False)
    except ImportError:
        print("⚠️  sounddevice non installé. Les chunks ne seront pas lus.")
    except Exception as e:
        print(f"⚠️  Erreur de lecture : {e}")

def save_audio_chunks(chunks, output_path):
    """
    Sauvegarde une liste de chunks audio dans un fichier WAV.
    """
    if not chunks:
        print("⚠️  Aucun chunk à sauvegarder.")
        return

    # Concaténer tous les chunks
    full_audio = torch.cat(chunks, dim=0)
    
    # Sauvegarder en WAV (convertir en int16 pour compatibilité)
    audio_numpy = full_audio.numpy()
    # Normaliser si nécessaire (Pocket-TTS produit généralement du float32 entre -1 et 1)
    if audio_numpy.dtype == np.float32:
        audio_numpy = (audio_numpy * 32767).astype(np.int16)
    
    scipy.io.wavfile.write(output_path, SAMPLE_RATE, audio_numpy)
    print(f"✅ Audio sauvegardé : {output_path}")

def process_with_progress_bar(chunks, total_estimated_chunks=None):
    """
    Affiche une barre de progression simple pendant la génération.
    """
    for i, chunk in enumerate(chunks):
        if total_estimated_chunks:
            progress = min(100, int((i + 1) / total_estimated_chunks * 100))
            bar = '█' * (progress // 5) + '░' * (20 - progress // 5)
            sys.stdout.write(f'\r🔊 Génération : [{bar}] {progress}% ({i+1}/{total_estimated_chunks} chunks)')
        else:
            # print("Not display chunk")
            sys.stdout.write(f'\r🔊 Génération : chunk {i+1} reçu')
        sys.stdout.flush()
        yield chunk
    print()  # Nouvelle ligne après la barre

# === SCRIPT PRINCIPAL ===

def main():
    parser = argparse.ArgumentParser(
        description="Génération audio en streaming avec Pocket-TTS"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="french_24l",
        help="Code de langue pour la synthèse (ex: fr, en, es, de, ...)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Bonjour, ceci est un test de génération audio en streaming avec Pocket TTS. Le texte peut être long, car le système va le générer par petits morceaux en temps réel.",
        help="Texte à synthétiser"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="fantine",
        help="Nom de la voix à utiliser (ex: fantine, alba-mackenna, ...)"
    )
    parser.add_argument(
        "--voice-file",
        type=str,
        default="tester3.wav",
        help="Chemin vers un fichier audio pour cloner une voix (remplace --voice)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_streaming.wav",
        help="Chemin du fichier WAV de sortie"
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Désactiver la lecture en temps réel"
    )
    parser.add_argument(
        "--frames-after-eos",
        type=int,
        default=None,
        help="Nombre de frames après EOS (défaut: auto)"
    )
    parser.add_argument(
        "--copy-state",
        action="store_true",
        default=True,
        help="Copier l'état de la voix (permet de la réutiliser)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température de génération (0.0-1.0). Plus élevée = plus diverse."
    )
    parser.add_argument(
        "--lsd-decode-steps",
        type=int,
        default=1,
        dest="lsd_decode_steps",
        help="Nombre d'étapes de génération. Plus élevé = meilleure qualité mais plus lent."
    )
    parser.add_argument(
        "--no-quantize",
        action="store_false",
        dest="quantize",
        default=True,
        help="Désactiver la quantification (activée par défaut)"
    )
    parser.add_argument(
        "--no-copy-state",
        action="store_false",
        dest="copy_state",
        help="Ne pas copier l'état (plus rapide mais non réutilisable)"
    )

    args = parser.parse_args()

    print("🚀 Démarrage du script de streaming Pocket-TTS")
    print("=" * 50)

    # === ÉTAPE 1 : Chargement du modèle ===
    print("📥 Chargement du modèle TTS...")
    start_time = time.time()
    try:
        tts_model = TTSModel.load_model()
        print(f"✅ Modèle chargé en {time.time() - start_time:.2f}s")
        print(f"   Taux d'échantillonnage : {tts_model.sample_rate} Hz")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return

    # === ÉTAPE 2 : Création de l'état vocal ===
    print(f"🎤 Création de l'état vocal pour : {args.voice_file or args.voice}")
    start_time = time.time()
    try:
        if args.voice_file:
            # Utiliser un fichier audio pour cloner la voix
            if not Path(args.voice_file).exists():
                print(f"❌ Fichier vocal non trouvé : {args.voice_file}")
                return
            voice_state = tts_model.get_state_for_audio_prompt(args.voice_file)
        else:
            # Utiliser une voix prédéfinie
            voice_state = tts_model.get_state_for_audio_prompt(args.voice)
        print(f"✅ État vocal créé en {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"❌ Erreur lors de la création de l'état vocal : {e}")
        return

    # === ÉTAPE 3 : Génération audio en streaming ===
    print(f"📝 Texte à synthétiser :")
    print(f"   \"{args.text[:100]}{'...' if len(args.text) > 100 else ''}\"")
    print(f"   (Longueur : {len(args.text)} caractères)")
    print("🎵 Génération en streaming...")
    print("-" * 50)

    # Estimation du nombre de chunks (approximatif)
    estimated_chunks = max(1, int(len(args.text) / 20))  # ~20 caractères par chunk

    # Stocker tous les chunks pour sauvegarde
    all_chunks = []
    chunk_count = 0
    first_chunk_time = None
    start_time = time.time()

    try:
        # Génération avec streaming
        stream_generator = tts_model.generate_audio_stream(
            model_state=voice_state,
            text_to_generate=args.text,
            frames_after_eos=args.frames_after_eos,
            copy_state=args.copy_state,
        )

        # Parcourir les chunks
        for audio_chunk in process_with_progress_bar(stream_generator, estimated_chunks):
            chunk_count += 1
            
            # Mesurer la latence du premier chunk
            if first_chunk_time is None:
                first_chunk_time = time.time()
                latency = first_chunk_time - start_time
                print(f"\n   ⏱️  Latence du premier chunk : {latency*1000:.0f} ms")
            
            # Ajouter au stockage
            all_chunks.append(audio_chunk)
            
            # Lecture en temps réel (si activée)
            if not args.no_play:
                play_audio_chunk(audio_chunk)
            
            # Afficher la taille du chunk (débogage)
            # print(f"   Chunk {chunk_count} : {audio_chunk.shape[0]} samples")
            
            # Simulation de temps réel : si on est trop rapide, attendre
            # pour simuler un flux réel (désactivé par défaut)
            # time.sleep(CHUNK_DURATION_MS / 1000.0)

    except KeyboardInterrupt:
        print("\n⏹️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération : {e}")
        import traceback
        traceback.print_exc()
        return

    # === ÉTAPE 4 : Résumé et sauvegarde ===
    total_time = time.time() - start_time
    total_audio_duration = sum(chunk.shape[0] for chunk in all_chunks) / SAMPLE_RATE

    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DE LA GÉNÉRATION")
    print(f"   Chunks générés : {chunk_count}")
    print(f"   Audio total : {total_audio_duration:.2f}s")
    print(f"   Temps total : {total_time:.2f}s")
    print(f"   Facteur temps réel : {total_audio_duration / total_time:.2f}x")
    
    if first_chunk_time:
        print(f"   Latence premier chunk : {(first_chunk_time - start_time)*1000:.0f} ms")

    # Sauvegarder le fichier audio complet
    if all_chunks and args.output:
        save_audio_chunks(all_chunks, args.output)

    print("\n✨ Streaming terminé !")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Interruption par l'utilisateur")
        sys.exit(0)
