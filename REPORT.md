# Hybrid Quantum-Classical Neural Network for Music Genre Classification

**Course:** Deep Learning & Data Analysis (Joint Term Project)
**Dataset:** Spotify Songs (`spotify_songs.csv`)
**Task:** Multi-class classification of `playlist_genre` — 6 classes: edm, latin, pop, r&b, rap, rock

---

## 1. Problem Definition

Given a set of audio features extracted from Spotify tracks, classify each song into one of 6 playlist genres. The novel contribution is embedding a Variational Quantum Circuit (VQC) inside a classical PyTorch model — a hybrid quantum-classical architecture — and comparing its performance against a purely classical baseline.

---

## 2. Dataset

- **Source:** `spotify_songs.csv` — 32,833 rows, 23 columns
- **After cleaning:** 29,792 rows (3,041 duplicates removed)
- **Features used (12):** danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms
- **Target:** `playlist_genre` (6 classes, approximately balanced)
- **Split:** Stratified train/val/test
  - Train: 21,449 | Val: 2,384 | Test: 5,959
  - `stratify=y` on both splits — class proportions preserved across all sets

---

## 3. Data Pipeline

### Step 1 — Preprocessing
1. Load and clean raw CSV (drop NaN targets, duplicates, fix key==-1 → 6)
2. Stratified train/val/test split
3. Scale all 12 features to **[-π, π]** with `MinMaxScaler` fitted on train only
   - [-π, π] chosen over [0, π]: gives RY gates access to both rotation directions on the Bloch sphere, producing richer quantum states and better class separability

### Step 2 — Feature Selection (replaced PCA)
- Initial approach was PCA (12→6 components)
- **PCA was dropped:** SVC benchmark showed a 13.8% accuracy drop (48.5% → 34.7%)
- Root cause: PCA maximises variance, not class discrimination
- **Fix:** Direct feature selection — top 6 features ranked by F-score + mutual information

| Feature | F-score | Mutual Info | Role |
|---|---|---|---|
| speechiness | 1054 | 0.123 | Separates rap/speech from music |
| danceability | 973 | 0.101 | Strong edm/latin signal |
| energy | 626 | 0.084 | Separates rock/edm from acoustic |
| instrumentalness | 363 | 0.082 | Vocal vs instrumental separation |
| tempo | 92 | **0.227** | Highest mutual info — rhythm is genre-defining |
| acousticness | 285 | 0.074 | Electronic vs acoustic separation |

- SVC on selected 6 features: **45.2%** (drop of only 3.3% from full 12 — within acceptable range)
- Note: The hybrid model no longer uses these Z arrays directly — its internal encoder learns the compression end-to-end

---

## 4. Architecture

### 4.1 Hybrid QNN (`HybridGenreClassifier`)

```
Input (batch, 12)   <- all 12 features, scaled to [-π, π]
        |
Classical Encoder:
  Linear(12->32) -> ReLU
  Linear(32->64) -> ReLU
  Linear(64->6)  -> Tanh x π     output in (-π, π), ready for RY gates
        |
Quantum VQC Layer:
  AngleEmbedding (RY gates, 6 qubits)
  StronglyEntanglingLayers (L=2, 36 trainable params)
  PauliZ measurements -> 6-dim vector in [-1, 1]
        |
Classical Decoder:
  BatchNorm1d(6)                  stabilises near-zero quantum outputs
  Linear(6->16) -> ReLU
  Linear(16->6)                   raw logits
        |
Output (batch, 6)
```

**Total parameters:** ~3,180  
**Quantum parameters:** 36 (2 layers × 6 qubits × 3 rotation angles)

**Key design rationale:**
- *Wide pre-encoder*: Classical layers before the quantum layer work with clean data — safe to expand to 32→64. The encoder learns which rotation angles best separate genres, replacing hand-picked feature selection.
- *Modest post-decoder*: QNN outputs are noisy (parameter-shift gradients). Expanding to only 16 neurons avoids amplifying quantum noise into high-dimensional space. Going to 32+ would let the model latch onto noise patterns.
- *Tanh×π encoding*: Maps encoder output to (-π, π), giving full bidirectional rotation range to all 6 qubits.
- *Gradient method*: `backprop` on simulator (required for gradient flow through classical encoder → QNN; `parameter-shift` fails with classical layers feeding the circuit — PennyLane #4462). Switch to `parameter-shift` for IBM hardware.

### 4.2 Classical Baseline (`ClassicalBaseline`)

```
Input (batch, 12)
  Linear(12->128) + ReLU + BatchNorm + Dropout(0.3)
  Linear(128->64) + ReLU + BatchNorm + Dropout(0.3)
  Linear(64->32)  + ReLU
  Linear(32->6)   <- raw logits
```

Dropout applied here (not in hybrid) because classical outputs are clean — dropout regularises without noise-amplification risk.

### 4.3 Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Feature scaling | [-π, π] | Full RY rotation range vs [0, π] half-range |
| Bottleneck | Feature selection | PCA dropped 13.8% accuracy; selection loses only 3.3% |
| Pre-QNN encoder | 12->32->64->6 + Tanh×π | Learns optimal angles end-to-end; wide = safe before noise |
| Post-QNN decoder | BN -> 6->16->6 | Modest expansion decodes QNN without noise amplification |
| Gradient (simulator) | backprop | Works with classical encoder → QNN gradient flow |
| Gradient (hardware) | parameter-shift | Required for IBM hardware; one-line switch in circuit.py |
| Loss | CrossEntropyLoss + raw logits | Avoids double-softmax (silent near-zero gradient bug) |
| Optimizer | Adam lr=0.001 | Lowered from 0.01 after observing noisy convergence |

---

## 5. Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Max epochs | 35 |
| Early stopping patience | 7 (on val_loss) |
| Device | CPU (default.qubit simulator) |

---

## 6. Results

### 6.1 Iteration History

Training was run multiple times with progressive improvements:

| Run | Architecture change | Hybrid val_acc | Baseline val_acc |
|---|---|---|---|
| Run 1 | Initial (PCA + [0,π] + Linear(6→6)) | 33.0% | 51.3% |
| Run 2 | Feature selection + [-π,π] scaling | 39.6% | 51.3% |
| Run 3 | Wide encoder (12→32→64→6) + decoder (6→16→6) + wider baseline | **48.7%** | **54.2%** |

### 6.2 Final Run — Hybrid QNN (35 epochs, no early stopping)

| Metric | Best (ep.31) | Final (ep.35) |
|---|---|---|
| Train loss | 1.345 | 1.325 |
| Val loss | **1.348** | 1.371 |
| Train acc | 48.5% | 48.9% |
| Val acc | **48.7%** | 46.6% |

Learning curve: Steady improvement from 20.4% (ep.1) to 48.7% (ep.31), showing the encoder learning meaningful quantum angle representations. No early stopping triggered — model still had forward momentum at ep.35.

### 6.3 Final Run — Classical Baseline (35 epochs, no early stopping)

| Metric | Best (ep.35) | Note |
|---|---|---|
| Train loss | **1.285** | last epoch |
| Val loss | **1.241** | last epoch — still improving |
| Train acc | 50.4% | |
| Val acc | **54.2%** | last epoch was best |

The baseline was still improving at epoch 35 (last epoch was best val_loss). More epochs with a learning rate scheduler would push this further.

### 6.4 Comparison

| Model | Val Accuracy | Val Loss | Parameters |
|---|---|---|---|
| Random baseline | 16.7% | — | 0 |
| **Hybrid QNN** | **48.7%** | **1.348** | ~3,180 |
| **Classical Baseline** | **54.2%** | **1.241** | ~22,000 |

**Gap: 5.5%** — down from 18.3% in Run 1.

---

## 7. Analysis

### Why the hybrid improved significantly (33% → 48.7%)

1. **Wide pre-encoder was the key unlock.** Feeding raw features directly into the quantum layer (Run 1/2) meant the QNN had to do both feature extraction AND classification. Adding a dedicated 12→32→64→6 encoder separates concerns: classical layers handle feature learning, quantum layer handles quantum processing.

2. **[-π, π] encoding improved state preparation.** Full bidirectional rotation gives each qubit access to both sides of the Bloch sphere. Features near the dataset mean map to ≈0 (ground state), while high/low outlier values rotate in opposite directions — creating more separable quantum states before the ansatz runs.

3. **Feature selection over PCA.** PCA's 13.8% accuracy drop indicated it was discarding discriminative structure. Feature selection preserved the actual values of the most genre-relevant features.

4. **Modest post-decoder.** Previous single Linear(6→6) had 42 weights to decode 6 quantum measurements. The 6→16→6 decoder gave enough capacity without amplifying quantum noise into a wide space.

### Why the classical baseline is harder to push past 54%

The dataset itself is the ceiling. These 12 features are high-level Spotify audio summaries (danceability=0.8, energy=0.6 etc.), not raw audio. Genre classification inherently requires signal-level features (mel spectrograms, MFCCs). Even optimal models on these 12 features likely top out at 60-65%. The baseline was still learning at ep.35 — more epochs and a learning rate scheduler would close the gap further.

### The hybrid vs classical story for the paper

The gap narrowed from **18.3% → 5.5%** across three training runs purely through architectural improvements — not by changing the quantum circuit itself. This demonstrates that the classical wrapper quality dominates hybrid model performance at the NISQ scale. The quantum layer's contribution is real but bounded by what it receives and how its outputs are decoded.

A 48.7% hybrid vs 54.2% classical — within 6% — on a noisy tabular classification task with only 36 quantum parameters is a reasonable NISQ-era result. The literature consistently shows VQCs underperform classical models on structured tabular data; the relevant metric is the margin, not the absolute accuracy.

---

## 8. Next Steps

- [ ] Run `scripts/run_evaluation.py` — test set evaluation, confusion matrices, per-class metrics
- [ ] Add learning rate scheduler (`ReduceLROnPlateau`) to trainer for next run
- [ ] Increase epochs to 50 for baseline (still improving at ep.35)
- [ ] Optional: IBM hardware run — switch `diff_method` to `parameter-shift`, set `use_ibm_hardware=True`
- [ ] Write final report comparison section using Section 6.4

---

*Last updated: 2026-04-14 — Run 3 complete. Best: Hybrid 48.7%, Classical 54.2%. Gap: 5.5%.*
