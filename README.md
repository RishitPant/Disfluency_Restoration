# Automatic Disfluency Restoration

A multimodal deep learning system that predicts the original disfluent Hindi transcript from a clean text input and its corresponding audio — built for the NPPE-2 Kaggle competition as part of a Deep Learning course.

---

## Problem Statement

Human speech is naturally filled with hesitations, filled pauses, and repetitions — "अं", "हम्म", "मतलब" — that ASR systems and text pipelines typically strip away. This project tackles the inverse problem: given a *cleaned* transcript and the original audio, restore the disfluencies to reconstruct what the speaker *actually* said, verbatim.

This sits at the intersection of **Automatic Speech Recognition (ASR)** and **Natural Language Generation (NLG)**, requiring the model to understand not just meaning, but the texture of spoken language.

**Evaluation Metric:** Word Error Rate (WER) — lower is better.

---

## Approach

### Multimodal Fusion via Token Injection

The core idea is to give the seq2seq model two complementary signals:

- **Clean text** — the grammatically correct, disfluency-free version of what was said
- **ASR transcript** — Whisper's raw transcription of the audio, which naturally preserves disfluencies

These are fused into a single input string using a custom separator token:

```
clean_text [ASR] whisper_transcription </s> <2en>
```

The seq2seq model (IndicBART) is then fine-tuned to map this combined input to the original disfluent target transcript. The `[ASR]` token is added as a new special token, with model embeddings resized accordingly.

### Why these models?

| Component | Model | Reason |
|-----------|-------|--------|
| ASR | `collabora/whisper-medium-hindi` | State-of-the-art Hindi ASR; preserves disfluencies better than generic multilingual models |
| Seq2Seq | `ai4bharat/IndicBART` | Pre-trained on Indic language corpora; superior Hindi tokenisation and language priors compared to mBART or T5 |

---

## Pipeline

```
Audio (.wav)
    │
    ▼
Whisper Medium (Hindi)
    │  ASR Transcription (cached to JSON)
    ▼
┌─────────────────────────────────────────────┐
│  "clean text [ASR] asr_text </s> <2en>"     │  ← Multimodal input
└─────────────────────────────────────────────┘
    │
    ▼
IndicBART Fine-tuning
    │  Target: "<2hi> disfluent_transcript </s>"
    ▼
Predicted Disfluent Transcript
```

---

## Data Preparation

The training data only provides the disfluent transcript. Clean labels are constructed programmatically by removing words listed in `unique_disfluencies.csv`:

```python
def remove_disfluencies(text, disfluencies):
    words = text.split()
    return " ".join([w for w in words if w not in disfluencies]).strip()

train_df["clean_transcript"] = train_df["transcript"].apply(
    lambda x: remove_disfluencies(x, disfluencies)
)
```

This clean transcript, paired with the original disfluent transcript, forms the training pairs for the seq2seq model.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `ai4bharat/IndicBART` |
| ASR model | `collabora/whisper-medium-hindi` |
| Batch size | 4 (effective 16 with gradient accumulation) |
| Gradient accumulation steps | 4 |
| Epochs | 5 |
| Learning rate | 5e-5 |
| Scheduler | Linear warmup (10% of total steps) |
| Max input/target length | 512 tokens |
| Optimiser | AdamW |
| Gradient clipping | 1.0 |
| Inference beam size | 8 |

**Best model checkpointing** — model weights are saved whenever validation loss improves across epochs.

**ASR caching** — Whisper transcriptions are cached to a JSON file after the first run to avoid redundant GPU inference on subsequent runs.

---

## Project Structure

```
├── Disfluency_Restoration.ipynb   # Main notebook
├── asr_cache/
│   └── asr_cache.json             # Cached Whisper transcriptions
├── best_model.pth                 # Saved model weights
└── submission.csv                 # Final predictions
```

**Dataset (from Kaggle input):**
```
├── train.csv                      # Disfluent transcripts + audio IDs
├── test.csv                       # Clean transcripts + audio IDs
├── unique_disfluencies.csv        # List of disfluent words/phrases
└── downloaded_audios/             # .wav audio files named by ID
```

---

## Dependencies

```
torch
transformers
librosa
tqdm
scikit-learn
pandas
numpy
jiwer          # for WER evaluation
```

Install with:

```bash
pip install torch transformers librosa tqdm scikit-learn pandas numpy jiwer
```

This notebook is designed to run on **Kaggle** (GPU T4 x2 recommended). To run locally, update `INPUT_DIR` and `AUDIO_DIR` paths and ensure a CUDA-enabled GPU is available.

---

## Key Design Decisions

**Gradient accumulation** simulates a larger effective batch size (16) within Kaggle's GPU memory constraints, which stabilises seq2seq fine-tuning.

**Custom `[ASR]` special token** explicitly signals to the model where the text modality ends and the audio-derived signal begins, allowing it to learn modality-specific attention patterns.

**ASR as a soft audio signal** — rather than feeding raw audio features directly into IndicBART (which has no audio encoder), Whisper acts as an intermediate transducer, converting audio into text that IndicBART can natively process. This avoids complex cross-modal attention engineering while still incorporating audio information.

**Forced BOS token (`<2hi>`)** during inference ensures the model generates in Hindi rather than defaulting to another language supported by IndicBART.

---

## Limitations & Future Improvements

- **Validation metric mismatch** — the model is checkpointed on validation loss, but the competition metric is WER. Future work would compute WER on a held-out validation sample after each epoch (using `jiwer`) and checkpoint on that directly.
- **Disfluency removal is exact-match only** — the data preparation step does not handle case variations or normalisation before matching against the disfluency list. A regex-based or unicode-normalised approach would be more robust for Hindi text.
- **No explicit audio-text alignment** — the fusion is positional (concatenation), not attention-based. A cross-attention mechanism that aligns audio timestamps with text positions could improve recall of disfluencies that occur at specific points in the utterance.
- **Single modality fallback** — when ASR output is empty (missing or unreadable audio), the model falls back to text-only input gracefully via the `build_multimodal()` function.

---

## Real-world Applications

- **Legal & medical transcription** — Verbatim records must capture hesitations and filled pauses exactly as spoken; cleaned transcripts are insufficient for evidentiary or clinical purposes
- **ASR training data augmentation** — Generating synthetic disfluent variants of clean text to improve robustness of downstream ASR models
- **Conversational AI evaluation** — Benchmarking voice assistants requires knowing what the user *actually* said, not what the ASR silently cleaned up
- **Accessibility & representation** — Faithfully representing speech patterns of users who stutter or have natural speech dysfluencies, rather than normalising them away
