# Bayesian Inference for Clinical Audio Enhancement

Master's coursework project for *Information Theory and Inference* (Physics of Data, University of Padua), with industry partner **Vounded**.

A Bayesian pre-processing layer that recovers a latent clean speech signal $x_t$ from a noisy single-microphone observation $z_t = x_t + n_t$ recorded in a clinical setting, then validates the enhancement by measuring downstream Word Error Rate (WER) on a real STT engine (AssemblyAI).

The hypothesis under test: **maximising mutual information $I(x;\hat{x})$ at the pre-processing stage correlates with lower WER**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

Each notebook produces artefacts in `data/synth/` consumed by the next one. Execute them in order from `notebooks/`:

| Phase | Notebook | What it does |
|---|---|---|
| 0 | `phase0_dataset.ipynb` | Synthesise the SNR-mixed dataset (−5, 0, +5, +10, +15 dB) |
| 1 | `phase1_kalman.ipynb` | Frame-adaptive AR(p) Kalman filter |
| 2 | `phase2_particle.ipynb` | SIR particle filter with Student-t likelihood |
| 3 | `phase3_info.ipynb` | KSG mutual information + STFT KL divergence |
| 4 | `phase4_wer_assemblyai.ipynb` | WER from AssemblyAI transcriptions |

## Repository layout

```
Audios/                            Raw inputs (Doctor, Patient, Background_Noise)
src/                               Functional Python modules
  audio_io.py                      Load/resample/mix utilities (16 kHz mono)
  kalman.py                        AR Kalman filter
  particle.py                      SIR particle filter
  info_theory.py                   KSG MI, STFT KL, surprisal
  wer.py                           jiwer wrapper
notebooks/                         One notebook per phase
data/synth/                        Generated figures, JSON metrics, audio samples
  *.png                            Plots used in the presentation
  *_results.json                   Numeric results (WER, MI, KL, SNR)
  transcription_segments_60s/      60s audio clips (30s–90s of original)
                                   + transcriptions.md (AssemblyAI output)
requirements.txt
```

The large WAVs in `data/synth/` (full-length mixes and per-SNR filter outputs) are **regenerated** by running phases 0–2 — they are not tracked in git. The 60-second segments in `data/synth/transcription_segments_60s/` are kept for direct listening.

## Conventions

- 16 kHz mono throughout.
- AR order = 10, frame = 320 samples (20 ms), hop = 160 samples (50% overlap).
- Information-theoretic metrics computed in **bits** (KSG estimator natively returns nats; results are converted).
