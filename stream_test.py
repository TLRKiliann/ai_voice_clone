#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de démonstration pour le streaming audio avec Pocket-TTS.
Utilise generate_audio_stream() pour une latence minimale.
Charge une voix clonée depuis un fichier .wav et sauvegarde l'état vocal en .safetensors.
"""

import time
import numpy as np
from pocket_tts import TTSModel
import scipy.io.wavfile

# --- Configuration ---
TEXT = "Bonjour, ceci est une démonstration du streaming audio avec ma voix clonée."
WAV_FILE = "test.wav"           # Fichier source pour le clonage vocal
STATE_FILE = "rien.safetensors" # Fichier pour sauvegarder l'état vocal
OUTPUT_WAV = "sortie_streaming.wav" # Fichier audio final généré

def main():
    print("--- Démonstration Streaming Pocket-TTS ---")
    print(f"Chargement du modèle Pocket-TTS...")
    tts_model = TTSModel.load_model(language="french_24l", temp=0.5)
    print(f"Modèle chargé. Taux d'échantillonnage : {tts_model.sample_rate} Hz")

    # --- 1. Chargement ou création de l'état vocal ---
    print(f"\n--- Étape 1 : État vocal ---")
    try:
        # Tentative de chargement de l'état vocal depuis le fichier .safetensors
        voice_state = tts_model.get_state_for_audio_prompt(STATE_FILE)
        print(f"État vocal chargé depuis '{STATE_FILE}'.")
    except Exception as e:
        print(f"Impossible de charger '{STATE_FILE}'. Création de l'état depuis '{WAV_FILE}'...")
        if not os.path.exists(WAV_FILE):
            print(f"ERREUR : Le fichier '{WAV_FILE}' est introuvable.")
            return

        # Création de l'état vocal à partir du fichier .wav
        voice_state = tts_model.get_state_for_audio_prompt(WAV_FILE)
        print(f"État vocal créé depuis '{WAV_FILE}'.")

        # Sauvegarde de l'état vocal pour un usage futur
        # Note: la sauvegarde directe d'un state n'est pas une fonction standard de la lib.
        # Nous sauvegardons le tenseur 'embedding' qui est l'essentiel du state.
        try:
            # La méthode recommandée est d'utiliser la CLI `pocket-tts export-voice`
            # ou de sauvegarder manuellement le tenseur d'embedding.
            # Récupération de l'embedding depuis l'objet state (c'est un dictionnaire)
            if hasattr(voice_state, 'embedding'):
                import safetensors.torch
                safetensors.torch.save_file({"embedding": voice_state.embedding}, STATE_FILE)
                print(f"État vocal sauvegardé dans '{STATE_FILE}'.")
            else:
                print("AVERTISSEMENT: Impossible de sauvegarder l'état vocal (objet inconnu).")
        except Exception as save_err:
            print(f"AVERTISSEMENT: Sauvegarde de l'état vocal échouée : {save_err}")

    # --- 2. Génération audio en streaming ---
    print(f"\n--- Étape 2 : Streaming audio ---")
    print(f"Texte : '{TEXT}'")

    audio_chunks = []
    chunk_count = 0
    first_chunk_time = None
    start_time = time.time()

    # Boucle de streaming : chaque chunk est reçu dès qu'il est généré
    for audio_chunk in tts_model.generate_audio_stream(voice_state, TEXT):
        if first_chunk_time is None:
            first_chunk_time = time.time()
            latency = (first_chunk_time - start_time) * 1000
            print(f"✓ Premier chunk reçu après {latency:.1f} ms")

        # Conversion du tenseur PyTorch en numpy pour traitement
        chunk_np = audio_chunk.numpy() if hasattr(audio_chunk, 'numpy') else np.array(audio_chunk)
        audio_chunks.append(chunk_np)

        chunk_count += 1
        chunk_duration = len(chunk_np) / tts_model.sample_rate
        # Affichage toutes les 10 itérations pour éviter de surcharger la console
        if chunk_count % 10 == 0:
            print(f"  Chunks reçus : {chunk_count}, durée cumulée : {chunk_count * 0.08:.2f}s")

    # --- 3. Sauvegarde du fichier audio final ---
    print(f"\n--- Étape 3 : Sauvegarde ---")
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        scipy.io.wavfile.write(OUTPUT_WAV, tts_model.sample_rate, full_audio.astype(np.int16))
        print(f"Fichier audio sauvegardé : '{OUTPUT_WAV}'")
        elapsed_time = time.time() - start_time
        audio_duration = len(full_audio) / tts_model.sample_rate
        rtf = audio_duration / elapsed_time if elapsed_time > 0 else 0
        print(f"Statistiques :")
        print(f"  - Nombre total de chunks : {chunk_count}")
        print(f"  - Durée audio : {audio_duration:.2f}s")
        print(f"  - Temps de génération : {elapsed_time:.2f}s")
        print(f"  - Facteur temps-réel : {rtf:.2f}x")
        print(f"  - Taille du fichier : {len(full_audio)} échantillons")
    else:
        print("ERREUR : Aucun chunk audio reçu.")

if __name__ == "__main__":
    import os
    # Vérification des dépendances
    try:
        import safetensors
    except ImportError:
        print("AVERTISSEMENT: 'safetensors' n'est pas installé.")
        print("       Installez-le avec : pip install safetensors")
        print("       La sauvegarde de l'état vocal en .safetensors sera désactivée.")
    main()