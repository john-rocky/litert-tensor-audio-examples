"""Enroll a voice: reference audio (~3-10 s) -> 1024-d x-vector .npy.

Runs the Qwen3-TTS ECAPA speaker encoder in PyTorch on the LOCAL checkpoint
(the 76-tensor speaker_encoder is in model.safetensors) — a local-path
variant of the production extract_speaker_embedding.py, so no HF download.
The .npy drops into audio_smoke.py --spk / build_real_prompt(spk=...).

Run with the qwen_tts venv:
  ~/qtts/bin/python enroll_speaker.py <checkpoint_dir> <ref.wav> <out.npy>
"""

import sys

import librosa
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


def main():
    ckpt, wav_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    wrapper = Qwen3TTSModel.from_pretrained(ckpt, device_map='cpu',
                                            dtype=torch.float32)
    audio, sr = librosa.load(wav_path, sr=24000, mono=True)
    emb = wrapper.model.extract_speaker_embedding(
        audio=audio.astype(np.float32), sr=sr)
    emb = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
    assert emb.shape == (1024,), emb.shape
    np.save(out_path, emb)
    print(f'saved {out_path}: shape {emb.shape} '
          f'L2 {np.linalg.norm(emb):.3f} (finite={np.isfinite(emb).all()})')


if __name__ == '__main__':
    main()
