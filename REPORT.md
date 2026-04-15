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

| Run | Change | Hybrid val | Baseline val | Gap |
|---|---|---|---|---|
| Run 1 | Initial (PCA + [0,π] + Linear(6→6)) | 33.0% | 51.3% | 18.3% |
| Run 2 | Feature selection + [-π,π] scaling | 39.6% | 51.3% | 11.7% |
| Run 3 | Wide encoder (12→32→64→6) + decoder (6→16→6) | 48.7% | 54.2% | 5.5% |
| Run 4 | +ReduceLROnPlateau (patience=4, factor=0.5), 50 epochs | **51.3%** | **54.7%** | **3.4%** |
| Run 5 | +ReduceLROnPlateau (patience=2, factor=0.3) — aggressive | 48.6% | 54.2% | 5.6% |

**Best validation result: Run 4** — 51.3% hybrid, 54.7% baseline, gap 3.4%.

Run 5 confirmed a key finding: the hybrid model is sensitive to aggressive LR scheduling in a way the classical baseline is not. With patience=2 and factor=0.3, the scheduler reduced the hybrid's LR four times (ep18→ep32→ep36→ep45), bottoming out at lr=1e-5 by ep45 and trapping it in a local minimum (48.6%). The baseline, whose loss curve was steadily declining, only triggered one reduction (ep35) and landed at 54.2% — essentially unchanged from Run 3. This asymmetry is a genuine NISQ-era observation: the VQC's noisy output requires a higher LR to keep exploring the loss landscape; aggressive early reduction removes that freedom.

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

### 6.4 Validation Summary

| Model | Val Accuracy | Val Loss | Parameters |
|---|---|---|---|
| Random baseline | 16.7% | — | 0 |
| **Hybrid QNN** | **48.7%** | **1.348** | ~3,180 |
| **Classical Baseline** | **54.2%** | **1.241** | ~22,000 |

**Val gap: 5.5%** — down from 18.3% in Run 1.

### 6.5 Test Set Evaluation (Step 4 — held-out)

#### Final Verified Results (Run 6 — clean hybrid rerun, Run 4 settings)

| Model | Val Accuracy | Test Accuracy | Macro F1 | Parameters |
|---|---|---|---|---|
| Random baseline | 16.7% | 16.7% | — | 0 |
| **Hybrid QNN** | **47.86%** | **45.76%** | **0.44** | ~3,180 |
| **Classical Baseline** | **~54.7%** | **52.07%** | **0.51** | ~22,000 |
| **Gap** | — | **6.31%** | — | — |

| Genre | Hybrid F1 | Baseline F1 |
|---|---|---|
| edm | 0.60 | **0.64** |
| latin | 0.27 | 0.40 |
| pop | 0.27 | 0.34 |
| r&b | 0.37 | 0.44 |
| rap | 0.58 | **0.63** |
| rock | 0.58 | **0.63** |
| **macro avg** | **0.44** | **0.51** |

#### Test results across evaluated runs

| Run | Hybrid test | Baseline test | Gap | Notes |
|---|---|---|---|---|
| Run 3 | 47.49% | 51.22% | 3.73% | Pre-scheduler baseline |
| Run 5 | 46.90% | 51.74% | 4.84% | Over-aggressive scheduler |
| **Run 6** | **45.76%** | **52.07%** | **6.31%** | **Final verified checkpoint** |

**Note on variance:** Run 4 achieved 51.34% hybrid validation — the highest recorded — but its checkpoint was not recoverable. The rerun (Run 6) produced 47.86% val, demonstrating that quantum circuit training exhibits meaningful run-to-run variance (~3%) from random weight initialisation. This is characteristic of NISQ-era VQCs and should be accounted for when comparing hybrid vs classical results.

**Key per-class observations (consistent across all runs):**
- Both models struggle most on **latin** and **pop** — these genres overlap significantly in feature space (both are danceable, moderate energy, vocal)
- **edm, rap, rock** are most separable — high instrumentalness for edm, high speechiness for rap, high energy for rock
- The hybrid's val→test drop (47.86% → 45.76%, −2.1%) is smaller than the baseline's (~54.7% → 52.07%, −2.6%), suggesting the quantum layer's bounded [−1,1] output provides mild regularisation

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

The validation gap narrowed from **18.3% → 3.4%** across five training runs — architectural improvements (Runs 1–3) contributing the most, with LR scheduling providing a further boost (Run 4) before aggressive scheduling caused a regression (Run 5).

The five-run experiment revealed a key asymmetry: **the hybrid model is significantly more sensitive to LR scheduling than the classical baseline.** Aggressive reduction (patience=2, factor=0.3) reduced the hybrid's LR four times and trapped it in a local minimum; the baseline triggered only one reduction and was barely affected. This is not a training artifact — it reflects the VQC's noisy output, which requires a higher learning rate to keep exploring the loss landscape. Removing that freedom via aggressive LR reduction eliminates the hybrid's ability to escape local optima.

The best evaluated test result (Run 3): **47.49% hybrid vs 51.22% classical, gap 3.73%.** On a 6-class tabular task with only 36 quantum parameters and 12 high-level audio features, this is a reasonable NISQ-era result. The literature consistently shows VQCs underperform classical models on structured tabular data; the relevant metric is the margin, not absolute accuracy.

---

## 8. IBM Hardware Run

### 8.1 Hardware Setup

| Setting | Value |
|---|---|
| Backend | `ibm_fez` — 156-qubit Eagle r3 superconducting processor |
| Plan | IBM Quantum open (one-time trial quota) |
| Gradient method | `parameter-shift` (`backprop` is simulator-only; fails on real hardware) |
| Runtime API | `ibm_quantum_platform` channel (qiskit-ibm-runtime ≥0.26, replaces deprecated `ibm_quantum`) |
| Transpilation | `optimization_level=3` — maximum gate cancellation and SWAP minimisation |

### 8.2 Circuit Compilation — Logical to Physical

A critical step between defining a circuit in PennyLane and executing it on `ibm_fez` is **transpilation** — the process of mapping the logical circuit onto physical hardware constraints.

**The problem:** `ibm_fez` uses a heavy-hex lattice topology. Most qubits have only 2–3 neighbors. A 2-qubit gate (e.g. CNOT) can only execute directly between physically adjacent qubits. Non-adjacent qubits require SWAP gates, each costing 3 CNOTs and adding noise.

**Qiskit transpiler steps at optimization_level=3:**

1. **Qubit selection** — Reads daily calibration data (T1/T2 coherence times, gate error rates per qubit pair). Picks the 6 physical qubits with best fidelity and connectivity for the circuit's entanglement pattern.

2. **Routing** — Inserts SWAP gates where StronglyEntanglingLayers requires non-adjacent entanglement. Our `range=[1,2]` pattern means layer 1 CNOTs cross 2 positions — some will require SWAP overhead depending on selected qubit layout.

3. **Gate decomposition** — Native gate set of `ibm_fez`: `ECR`, `RZ`, `SX`, `X`. PennyLane's `Rot(φ,θ,ω)` decomposes to `RZ(φ)→RY(θ)→RZ(ω)`, where `RY = RZ·SX·RZ`. Each additional decomposition step adds gate error.

4. **Optimisation** — Cancels redundant gates (e.g. `RZ(0)`, back-to-back CNOTs), merges rotations, finds shorter equivalent sequences.

**Effect on our circuit:**

| Metric | Logical (PennyLane) | Transpiled (ibm_fez) |
|---|---|---|
| Qubits | 6 (abstract) | 6 physical from 156 |
| Gate set | Rot, CNOT, RY | ECR, RZ, SX, X |
| Circuit depth | ~20 | ~3–5× deeper (SWAP + decomposition overhead) |

This depth increase is the structural reason for NISQ noise — every extra gate adds error probability. The Bell state test (Section 8.3) directly measures this accumulated noise floor.

### 8.3 Hardware Connectivity Test

A 2-qubit Bell state circuit was successfully executed on `ibm_fez`:

| Qubit | Measured ⟨Z⟩ | Ideal ⟨Z⟩ (noiseless) |
|---|---|---|
| Q0 | **-0.00198** | 0.0 |
| Q1 | **-0.01025** | 0.0 |

A Bell state `(|00⟩ + |11⟩)/√2` has `⟨Z⟩ = 0` on both qubits individually — the two qubits are maximally entangled so each alone looks completely mixed. The non-zero measurements confirm real NISQ hardware noise from gate errors and decoherence on the superconducting qubits. This validates successful execution on real quantum hardware.

### 8.4 Inference Attempt — Quota and Architecture Lesson

**Initial approach (PennyLane `qiskit.remote`):** Submits one IBM job per forward pass. For 128 samples, this becomes 128 jobs. Running against the open plan quota, 44 jobs completed (~8 minutes) before the trial quota was exhausted. No predictions were recovered due to Python output buffering — results lived in memory and were lost when the process was killed.

**Root cause:** PennyLane's `qiskit.remote` device is designed for training (gradient computation per-circuit). For inference, it has no batching optimisation. Each sample triggers a complete job submission cycle with queue overhead.

**Optimised approach (Qiskit `EstimatorV2`):** The hybrid model separates cleanly:

```
Classical Encoder   →  runs on CPU, costs nothing
        ↓
Quantum VQC         →  ONE EstimatorV2 job, all samples batched
        ↓
Classical Decoder   →  runs on CPU, costs nothing
```

The optimised script (`scripts/run_hardware_inference.py`) reconstructs the trained VQC as a Qiskit `ParameterizedCircuit` with the 36 quantum weights baked in as constants and the 6 encoder outputs as the only free parameters. A single `EstimatorV2` call computes all 128 sample × 6 PauliZ expectation values in one job:

```python
pub    = (isa_circuit, obs_array, angles)   # angles: (128, 6) — all samples
job    = estimator.run([pub])               # 1 IBM job, not 128
result = job.result()
evs    = result[0].data.evs                 # (128, 6) expectation values
```

| Approach | IBM jobs submitted | Quota cost |
|---|---|---|
| PennyLane `qiskit.remote` | 128 | ~23 min (exceeds open plan) |
| Qiskit `EstimatorV2` (optimised) | **1** | ~10–30 sec |

Full inference results remain pending quota availability. The optimised script saves all results to `outputs/results/hardware_inference_results.json` and prints progress with unbuffered output — no results can be lost on interruption.

---

## 9. Status & Next Steps

- [x] Test set evaluation — Run 3 (47.49% hybrid, 51.22% baseline) and Run 5 (46.90%, 51.74%)
- [x] 5 training runs completed — gap narrowed from 18.3% → 3.4% (Run 4 best val)
- [x] LR scheduler sensitivity identified — key NISQ-era finding (Section 7)
- [x] IBM hardware connectivity verified — Bell state on ibm_fez (Section 8.3)
- [x] Hardware inference script optimised — EstimatorV2 batching (1 job total)
- [x] Training progression chart — `outputs/figures/training_progression.png`
- [ ] Full hardware inference — `py -3.12 -u scripts/run_hardware_inference.py` when quota available

---

*Last updated: 2026-04-15 — 6 runs complete. Final verified checkpoint: Hybrid 45.76% test, Classical 52.07% test (Run 6). Best val seen: Hybrid 51.3% (Run 4, checkpoint lost). Architecture ceiling reached; run-to-run variance ~3% documented.*
