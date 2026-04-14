# Hybrid Quantum-Classical Neural Network for Music Genre Classification

**Course:** Deep Learning & Data Analysis (Joint Term Project)
**Dataset:** Spotify Songs (`spotify_songs.csv`)
**Task:** Multi-class classification of `playlist_genre` — 6 classes: edm, latin, pop, r&b, rap, rock

---

## 1. Problem Definition

Given a set of audio features extracted from Spotify tracks, classify each song into one of 6 playlist genres. The novel contribution is embedding a Variational Quantum Circuit (VQC) inside a classical PyTorch model — a hybrid quantum-classical architecture — and comparing its performance against a purely classical baseline with the same parameter budget.

---

## 2. Dataset

- **Source:** `spotify_songs.csv`
- **Features used (12):** danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms
- **Target:** `playlist_genre` (6 classes, balanced)
- **Split strategy:** Stratified train/val/test split
  - Test: 20% of total data
  - Val: 10% of remaining (after test split)
  - Train: remainder
  - `stratify=y` used on both splits — each subset mirrors the full dataset's class proportions

---

## 3. Data Pipeline

### Step 1 — Preprocessing (`run_eda.py`)
1. Load raw CSV, validate columns
2. Clean data, separate features and target
3. Stratified train/val/test split
4. Scale features to **[0, π]** with `MinMaxScaler` (fitted on train only)
   - Range [0, π] is required for RY gate angle encoding
5. Encode labels with `LabelEncoder`
6. Save all processed arrays to `data/processed/`

### Step 2 — Bottleneck (`run_bottleneck.py`)
1. **PCA** compresses 12 features → 6 components (`PCAReducer`)
   - Fitted on train only, applied to val/test
2. PCA output re-scaled to **[0, π]** with a second `MinMaxScaler`
   - Necessary because PCA output is unbounded
3. Compressed arrays saved as `Z_train`, `Z_val`, `Z_test`

---

## 4. Architecture

### 4.1 Hybrid QNN (`HybridGenreClassifier`)

```
Input (batch, 6)   ← PCA-compressed, scaled to [0, π]
        │
   Quantum Layer   ← qml.qnn.TorchLayer wrapping VQC
        │
  BatchNorm1d(6)   ← stabilises near-zero quantum outputs (barren plateau)
        │
   Linear(6, 6)    ← raw logits, no Softmax
        │
  Output (batch, 6)
```

**VQC Circuit:**
- Encoding: `qml.AngleEmbedding` with RY gates — one qubit per feature, O(n) depth
- Ansatz: `qml.StronglyEntanglingLayers` (n_layers=2) — long-range entanglement
- Measurement: `qml.expval(qml.PauliZ(i))` on all 6 qubits → 6-dim vector in [-1, 1]
- Gradient method: **parameter-shift rule** — exact gradients, works on both simulator and real IBM hardware
- Trainable parameters: 2 × 6 × 3 = **36 quantum + 6 classical = 42 total**

### 4.2 Classical Baseline (`ClassicalBaseline`)

```
Input (batch, 12)  ← full 12 features, scaled to [0, π]
        │
  Linear(12, 16) + ReLU + BatchNorm1d(16)
        │
   Linear(16, 8) + ReLU
        │
    Linear(8, 6)   ← raw logits
        │
  Output (batch, 6)
```

- Intentionally constrained to a similar parameter budget for fair comparison
- **~330 parameters total**

### 4.3 Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Bottleneck | PCA 12→6 | Fast, interpretable; matches qubit count |
| Encoding | RY angle encoding | O(n) depth, continuous features, NISQ-friendly |
| Ansatz | StronglyEntanglingLayers L=2 | Long-range entanglement, proven template |
| Gradient | parameter-shift | Works identically on simulator and real hardware |
| Stabiliser | BatchNorm1d after quantum layer | Mitigates barren plateau zero-gradient collapse |
| Loss | CrossEntropyLoss with raw logits | Avoids double-softmax silent bug |
| Optimizer | Adam lr=0.01 | Handles noisy quantum gradients better than SGD |

---

## 5. Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping patience | 5 (on val_loss) |
| Device | CPU (default.qubit simulator) |

---

## 6. Results

### 6.1 Hybrid QNN

- **Early stopped at epoch 21** (no val_loss improvement for 5 consecutive epochs)
- Best checkpoint: epoch 16 (val_loss: 1.6065)

| Metric | Best (ep.16) | Final (ep.21) |
|---|---|---|
| Train loss | 1.618 | 1.616 |
| Val loss | **1.606** | 1.615 |
| Train acc | 33.0% | 32.9% |
| Val acc | **33.0%** | 32.8% |

**Learning curve summary:** Loss decreased very slowly from 1.699 → 1.606 over 21 epochs (~5.5% total reduction). Accuracy was nearly flat from epoch 2 onwards, hovering between 31–33%.

### 6.2 Classical Baseline

- **Ran all 30 epochs** (no early stopping triggered)
- Best checkpoint: epoch 28 (val_loss: 1.324)

| Metric | Best (ep.28) | Final (ep.30) |
|---|---|---|
| Train loss | 1.333 | 1.329 |
| Val loss | **1.324** | 1.336 |
| Train acc | 48.6% | 49.0% |
| Val acc | **50.5%** | 49.5% |

**Learning curve summary:** Loss dropped sharply in the first 2 epochs (1.587 → 1.423), then continued improving steadily. Accuracy climbed from 35.8% to ~49%.

### 6.3 Comparison

| Model | Val Accuracy | Val Loss | Params | Epochs |
|---|---|---|---|---|
| Random baseline | 16.7% | — | 0 | — |
| **Hybrid QNN** | **33.0%** | **1.606** | **42** | **21** |
| **Classical Baseline** | **50.5%** | **1.324** | **~330** | **30** |

---

## 7. Analysis

### Why the hybrid model underperformed

1. **Barren plateau:** Quantum expectation values cluster near zero, producing near-zero gradients throughout training. BatchNorm partially mitigates this but does not eliminate it. The nearly flat accuracy curve (31–33% across all 21 epochs) is a textbook barren plateau signature.

2. **Limited expressibility:** With only 6 qubits and 2 ansatz layers, the VQC has limited capacity to capture the complex decision boundaries between 6 genre classes.

3. **Information loss through PCA:** Compressing 12 features to 6 via PCA before the quantum layer discards variance that the classical baseline retains. The baseline operates on all 12 features directly.

4. **Noisy gradients:** The parameter-shift rule computes exact gradients but requires 2 circuit evaluations per parameter per sample, making training inherently slower and noisier than classical backpropagation.

### What both models achieved

- Both significantly outperform random guessing (16.7%)
- No overfitting observed in either model — train and val accuracy tracked closely throughout
- The hybrid model's 33% accuracy represents **2× better than random** with only 42 parameters

### Literature context

Underperformance of VQCs on classical tabular data vs. classical models is a widely documented finding in the quantum ML literature. NISQ-era quantum hardware and simulators face fundamental challenges (barren plateaus, limited qubit counts, noise) that prevent quantum models from competing with classical DNNs on structured data tasks at this scale. This result is consistent with and expected by the current state of the field.

---

## 8. Next Steps

- [ ] Run `scripts/run_evaluation.py` for full test set evaluation and confusion matrices
- [ ] Optional: Try autoencoder bottleneck instead of PCA — may preserve more structure
- [ ] Optional: IBM hardware run — expect 3–8% accuracy drop due to NISQ noise
- [ ] Write final report comparison section using results from Section 6.3

---

*Last updated: 2026-04-14 — Training complete (Steps 1–3 done)*
