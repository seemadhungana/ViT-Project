# Examining the Temporal Dynamics of Human Alignment by Probing Model Learning Windows

## Problem Statement & Overview

Computer vision research has increasingly focused on guiding deep neural networks (DNNs) toward learning more human-like representational structures. However, **little is known about when and how human representational alignment emerges during network learning**. Research shows that DNNs often exhibit nonlinear training dynamics, with rapid performance gains concentrated within brief "critical periods" of learning, but it remains unknown whether these patterns extend to a model's ability to learn human representations.

### Research Motivation

Drawing from **developmental neuroscience**, where critical periods are identified by observing differential sensitivity to environmental disruptions at different developmental stages, we apply **temporally localized perturbations** to probe when human alignment learning is most vulnerable or malleable.

### Key Research Questions

1. **Do critical periods exist for human alignment learning?** Unlike task performance (accuracy/loss), does behavioral alignment with human perception follow distinct temporal dynamics during training?

2. **When is alignment learning most sensitive to disruption?** By perturbing training at different epochs, can we identify windows of heightened plasticity or vulnerability for human-like representation learning?

3. **How do models recover from perturbations?** Does perturbation timing and duration affect whether models can recover human alignment, and do recovery dynamics differ from task loss recovery?

### Our Approach

We introduce **perturbations in the target embeddings** of a CLIP behavioral-alignment fine-tuning pipeline (CLIP-HBA-Behavior), substituting randomly-generated targets for ground-truth behavioral targets at specific epochs and for variable durations. This approach allows us to map the temporal structure of alignment learning by measuring:

- How disruptions affect task loss and human alignment **at the time of perturbation**
- How they affect **final model performance**
- How well the model **recovers** depending on perturbation timing and duration

Additionally, we extend this analysis to baseline **ViT models trained on ImageNet** (not fine-tuned for human alignment) to understand how behavioral alignment emerges naturally during standard supervised learning.

### Key Findings

Our results reveal **temporal asymmetries** in human representational learning:

- **Short, early perturbations transiently enhanced human alignment**, whereas longer or late-stage perturbations caused decreases
- In the **loss landscape**, early and early-middle perturbations negatively affected minimum test loss, whereas later perturbations had little effect
- Models **recovered faster from late-training perturbations**
- Importantly, **alignment and accuracy follow distinct temporal dynamics**, suggesting they arise through partially independent mechanisms

These findings suggest that optimization towards human alignment in deep networks may proceed through **temporally structured phases with distinct stability properties**, offering a new lens for analyzing inductive biases and training robustness.

---

## Methodology

### Techniques from Transformers Course

Our research applies and extends several core concepts from the Transformers course to investigate a fundamental scientific question about representation learning:

#### 1. Vision Transformers (ViT) & CLIP Architecture

We leverage two foundational vision transformer architectures:

**ViT-Base (patch16_224)** for baseline experiments:
- 12 transformer encoder blocks with 768-dimensional embeddings
- Processes images as sequences of 16×16 patches
- Standard supervised training on ImageNet (1000 classes)
- Used to study how behavioral alignment emerges naturally during classification training

**CLIP-ViT-L/14** for human alignment fine-tuning:
- Pretrained vision-language model with contrastive learning
- Larger architecture (ViT-Large) with 14×14 patches
- Fine-tuned specifically for human behavioral alignment using THINGS dataset
- Demonstrates transfer learning from multimodal pretraining to cognitive alignment

This connects to course concepts on **vision transformers**, **multimodal learning**, and **transfer learning**.

#### 2. Parameter-Efficient Fine-Tuning (PEFT)

Rather than fine-tuning all parameters of the large CLIP model, we apply **Domain-specific Rank Adaptation (DoRA)**:
- Only the last 2 vision transformer layers + 1 text layer are unfrozen
- Significantly reduces trainable parameters (~2.5M vs. full model)
- Enables efficient fine-tuning while preserving pretrained representations
- Learning rate: 3e-4, AdamW optimizer, batch size: 64

This applies course concepts on **adapter methods**, **LoRA/DoRA**, and **efficient fine-tuning strategies** that are central to modern transformer applications.

#### 3. Representational Similarity Analysis (RSA)

We measure human alignment using **RSA**, which quantifies similarity between representational geometries:

```
Behavioral Alignment = Spearman ρ(Model RDM, Human Behavioral RDM)
Neural Alignment = Spearman ρ(Model RDM, fMRI RDM)
```

**Representational Dissimilarity Matrices (RDMs):**
- Model RDM: Pairwise distances between model embeddings for 48 test images
- Behavioral RDM: Human triplet odd-one-out judgments from THINGS dataset
- Neural RDM: fMRI activation patterns from ventral visual cortex (V1-V4, LO, IT, FFC, PIT)

This connects to course discussions on **probing representations**, **interpretability**, and **evaluation beyond task metrics**.

#### 4. Critical Periods Framework

Our perturbation methodology draws from both **developmental neuroscience** and **machine learning research on training dynamics**:

**Biological Inspiration:**
- Critical periods in visual cortex development (Hubel & Wiesel, 1970)
- Heightened plasticity windows where experience shapes cortical organization
- Differential sensitivity to environmental disruption at different developmental stages

**Machine Learning Translation:**
- Early training characterized by high sensitivity to parameter updates (Achille et al., 2019)
- Regularization timing affects final generalization (Golatkar et al., 2019)
- Networks memorize simple patterns before overfitting to noise (Zhang et al., 2017)

**Our Innovation:** We extend this framework to **human alignment learning**, not just task performance.

#### 5. Perturbation Experiment Design

We systematically test four perturbation types:

1. **Random Target Noise** (primary focus): Replace ground-truth 66D behavioral embeddings with randomly generated tensors from the same distribution
2. **Label Shuffling**: Randomly permute target embeddings across training samples
3. **Gaussian Image Noise**: Add Gaussian noise to input images (ε = 0.1)
4. **Blank Images**: Replace inputs with uniform gray images (pixel value = 0.5)

**Experimental Paradigms:**

- **Single-Epoch Perturbation Sweep**: Perturb exactly one epoch at each point during training (epochs 1-98), then resume normal training
  - Measures immediate vulnerability at each training stage
  - Reveals when alignment learning is most sensitive

- **Variable-Length Perturbations**: Test perturbations starting at epochs [1, 2, 3, 6, 7, 8, 10, 70, 80, 90] for durations [5, 10, 20, 30, 40, 50] epochs
  - Maps interaction between timing and duration
  - 180+ unique experimental conditions

This demonstrates course concepts on **training dynamics**, **robustness analysis**, and **controlled experimentation** in deep learning.

#### 6. Training & Optimization

**CLIP-HBA-Behavior Training:**
- Optimizer: AdamW (lr=3e-4, weight decay enabled)
- Loss: MSE between predicted and target 66D embeddings
- Epochs: 500 maximum with early stopping (patience=20)
- Dataset: 1,806 THINGS images (80/20 train/val split) + 48 held-out test images

**ViT Baseline Training:**
- Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
- Loss: Cross-entropy for ImageNet classification
- LR Schedule: Cosine annealing with linear warmup
- Epochs: 100
- Distributed training with DDP and mixed precision (AMP)

**Recovery Dynamics Measurement:**
- Recovery = first epoch after perturbation where test loss returns within 1% of baseline
- Models that never recover are marked "Never Recovered" (NR)
- Separate tracking for loss recovery vs. alignment recovery

### Connection to Course Content

This project demonstrates research-oriented applications of transformers:

- **Scientific Investigation**: Not just achieving high accuracy, but understanding *when* and *how* representations develop
- **Multimodal Learning**: CLIP's vision-language pretraining as foundation for behavioral alignment
- **Transfer Learning**: From contrastive pretraining → human perception fine-tuning
- **Training Dynamics**: Temporal analysis of learning reveals phase-like transitions
- **Robustness**: Perturbations as probes for understanding representational stability

The project bridges **computer vision**, **cognitive neuroscience**, and **developmental learning theory**—demonstrating how transformer research can address fundamental questions about human-AI alignment.

---

## Implementation & Demo

### Project Structure

```
ViT-Project/
├── Training/                           # Training scripts and experiments
│   ├── vit_training/
│   │   ├── baseline/
│   │   │   └── train_vit_sgd.py       # ViT-Base ImageNet training
│   │   └── single_epoch/
│   │       └── measure_single_epoch_perturbation_effect.py
│   ├── clip_behavioral_finetuning/
│   │   ├── baseline/
│   │   │   └── clip_train_behavior_baseline.py
│   │   ├── uniform_sweep/
│   │   │   └── clip_train_behavior_sweep.py
│   │   └── length_experiments/
│   │       └── clip_train_behavior_lengths.py
│   └── functions/
│       ├── cvpr_train_behavior_things_pipeline_baseline.py
│       └── spose_dimensions.py         # 66D semantic dimension definitions
├── Data/                               # Data and results
│   ├── vit_results/
│   │   └── rsa_results_final.csv      # ViT baseline RSA scores
│   └── clip_results/
│       ├── baseline_clip_results_seed1.csv
│       ├── single_sweep_experiments/   # Single-epoch perturbation sweeps
│       ├── image_noise/                # Gaussian noise perturbations
│       ├── label_shuffle/              # Label shuffling experiments
│       ├── target_noise/               # Random target perturbations
│       └── perturb_length_experiments/ # Variable-length perturbations
└── Figures/                            # Analysis notebooks
    ├── fig1/                          # Baseline behavioral alignment
    ├── fig2/                          # Perturbation type comparison
    ├── fig3/                          # Single-epoch sweep results
    └── fig4/                          # Recovery dynamics
```

### Key Scripts & Experiments

#### 1. Baseline CLIP-HBA Training ([Training/clip_behavioral_finetuning/baseline/clip_train_behavior_baseline.py](Training/clip_behavioral_finetuning/baseline/clip_train_behavior_baseline.py))

Establishes the unperturbed baseline for behavioral alignment:

```python
# Configuration
model = CLIP-ViT-L/14 (pretrained)
adaptation = DoRA(last_2_vision_layers + 1_text_layer)
optimizer = AdamW(lr=3e-4)
loss = MSE(predicted_embedding, target_66d_behavioral_embedding)
epochs = 500 (early_stopping_patience=20)
dataset = THINGS (1806 train, 48 test images)
```

**Training Dynamics:**
- Exhibits **S-shaped learning curve** for behavioral alignment
- RSA improves from 0.46 (epoch 1) → 0.71+ (epochs 10-14)
- **"Elastic learning" window** around epochs 5-10: small loss improvements yield large alignment gains
- Early stopping typically occurs at epoch ~15 (minimum test loss)

**Results:** See [Data/clip_results/baseline_clip_results_seed1.csv](Data/clip_results/baseline_clip_results_seed1.csv)

**Figure 1a: CLIP-HBA Baseline Behavioral Alignment**

![CLIP-HBA Baseline](Figures/fig1%20(Baseline%20CLIP-HBA%20Behavioral%20Alignment)/fig1a.png)

**Figure 1b: ViT Baseline Behavioral Alignment**

![ViT Baseline](Figures/fig1%20(Baseline%20CLIP-HBA%20Behavioral%20Alignment)/fig1b.png)

#### 2. Perturbation Type Comparison ([Training/clip_behavioral_finetuning/](Training/clip_behavioral_finetuning/))

Tests four different perturbation types at selected epochs to understand which disruptions are most impactful:

```python
# Perturbation types tested
1. Image Noise (Gaussian): Add Gaussian noise to input images
2. Blank Image: Replace images with uniform gray
3. Label Shuffle: Randomly permute target embeddings
4. Target Noise: Replace targets with random embeddings (primary focus)

# Epochs tested: [5, 15, 25, 35, 45, 70, 98]
```

**Key Findings (Figure 2):**

**CLIP-HBA Results:**
- **Target noise** shows strongest effects at all epochs
- **Label shuffle** causes consistent degradation across all time points
- **Image noise and blank image** have moderate, stable effects
- All perturbation types show increasing impact as training progresses (epoch 5 < epoch 98)

**ViT Results:**
- All perturbation types cause large increases in validation loss
- Effects are relatively uniform across perturbation types
- RSA degradation is substantial across all conditions tested
- **Critically**: ViT shows consistent degradation across all perturbation timings, unlike CLIP-HBA

**Interpretation:** Different perturbation types have varying impacts, with target-level disruptions (target noise, label shuffle) being more destructive than input-level disruptions (image noise, blank images). Importantly, ViT and CLIP-HBA show fundamentally different temporal dynamics in response to perturbations.

**Results:** See [Data/clip_results/image_noise/](Data/clip_results/image_noise/), [Data/clip_results/label_shuffle/](Data/clip_results/label_shuffle/), [Data/clip_results/uniform_target/](Data/clip_results/uniform_target/), [Data/clip_results/target_noise/](Data/clip_results/target_noise/)

**Figure 2a: CLIP-HBA Perturbation Type Comparison**

![CLIP Perturbation Types](Figures/fig2%20(Effects%20of%20Different%20Perturbations)/fig2a.png)

**Figure 2b: ViT Perturbation Type Comparison**

![ViT Perturbation Types](Figures/fig2%20(Effects%20of%20Different%20Perturbations)/fig2b.png)

#### 3. Single-Epoch Perturbation Sweep ([Training/clip_behavioral_finetuning/uniform_sweep/clip_train_behavior_sweep.py](Training/clip_behavioral_finetuning/uniform_sweep/clip_train_behavior_sweep.py))

Tests immediate effects of perturbing exactly one epoch across all 98 training epochs:

```python
# Experimental protocol
for perturb_epoch in range(1, 99):
    1. Load baseline checkpoint from epoch (perturb_epoch - 1)
    2. Load baseline random state for reproducibility
    3. Apply random target perturbation for exactly 1 epoch
    4. Resume normal training for remaining epochs
    5. Measure Δ_loss and Δ_alignment relative to baseline at perturbation epoch
```
**Figure 3a: Single-Epoch Perturbation Effects on Test Loss**

![Single-Epoch Sweep Loss](Figures/fig3%20(Single%20Sweep%20Perturbation%20Experiments)/fig3a.png)

**Figure 3b: Single-Epoch Perturbation Effects on Behavioral Alignment**

![Single-Epoch Sweep Alignment](Figures/fig3%20(Single%20Sweep%20Perturbation%20Experiments)/fig3b.png)

**Key Findings (Figure 3):**

**CLIP-HBA Behavioral Alignment (Δ RSA):**
- **Early epochs (1-20)** show **positive deviations** (green bars): perturbations improve alignment!
  - Peak improvement around epochs 5-10
  - Counterintuitive result: brief noise enhances human-like representations
- **Late epochs (60-98)** show **negative deviations** (blue bars): perturbations degrade alignment
  - Progressive worsening as training advances
  - Suggests representations "lock in" during late training

**CLIP-HBA Test Loss (Δ Test Loss):**
- **Early epochs** show minimal loss increases (red bars barely visible)
- **All epochs** eventually show positive deviations (perturbations increase loss)
- Loss impact is more uniform across epochs than alignment impact

**Model Comparison:**
- **CLIP-HBA exhibits a reversal**: early perturbations help alignment, late perturbations hurt it
- **ViT does NOT show this reversal**: perturbations consistently degrade alignment regardless of timing
- This suggests the alignment enhancement from early perturbations is specific to the fine-tuning regime, not a universal property of vision transformers

**Interpretation:**
- Alignment and accuracy have **different critical periods** (in CLIP-HBA)
- Early training is plastic for alignment but minimally affected for loss (CLIP-HBA specific)
- Late training is fragile for alignment but loss can still be recovered
- **ViT's uniform degradation** suggests standard supervised learning lacks the temporal asymmetry seen in behavioral fine-tuning

**Results:** See [Data/clip_results/single_sweep_experiments/](Data/clip_results/single_sweep_experiments/) (90+ training runs)



#### 4. Variable-Length Perturbation Recovery ([Training/clip_behavioral_finetuning/length_experiments/clip_train_behavior_lengths.py](Training/clip_behavioral_finetuning/length_experiments/clip_train_behavior_lengths.py))

Systematically varies perturbation timing and duration to study recovery dynamics:

```bash
# Command-line interface
python clip_train_behavior_lengths.py \
  --perturb_type random_target \
  --perturb_epoch 10 \
  --perturb_length 20 \
  --perturb_seed 0 \
  --random_seed 1 \
  --baseline_dora_directory ../baseline/dora_params/ \
  --baseline_random_state_path ../baseline/random_state_epoch10.pkl \
  --baseline_split_indices_path ../baseline/split_indices.pkl \
  --output_dir output/random_target_e10_l20/ \
  --output_base_directory ./output/
```

**Parameter Space (Figure 4):**
- Start epochs: [1, 2, 3, 6, 7, 8, 10, 40, 70, 80, 90]
- Perturbation lengths: [2, 5, 10, 20, 30, 40, 50] epochs
- Total: 136 unique (epoch, length) combinations tested
- Recovery metric: Epochs to return within 1% of baseline test loss

**Key Findings (Figure 4):**

**Recovery Time Patterns:**
- **Early perturbations (epochs 1-3)** show moderate recovery times (10-60 epochs)
  - Longer perturbations require longer recovery
  - Most eventually recover successfully

- **Middle perturbations (epochs 6-10)** show longest recovery times and some failures
  - Several combinations **NEVER RECOVER** (marked "NR" in red)
  - These represent permanent developmental disruptions
  - Critical window where perturbations have lasting damage

- **Late perturbations (epochs 40, 70, 80, 90)** recover very quickly (5-20 epochs)
  - Fast recovery even for long perturbations (length 50)
  - Suggests late training occupies flatter, more stable loss regions

**Never Recovered (NR) Conditions:**
Specific (epoch, length) combinations that never returned to within 1% of baseline:
- Models perturbed during critical middle period that could not recover despite continued training
- Indicates permanent trajectory alteration during sensitive developmental window

**Interpretation:**
- **Temporal asymmetry in resilience**: Late training is more resilient than middle training
- **Critical vulnerability window**: Epochs 6-10 are most fragile to sustained perturbations
- **Recovery speed inversely correlates with perturbation timing**: Later = faster recovery
- **Loss landscape interpretation**: Early/middle training explores narrow valleys (hard to recover), late training occupies broad basins (easy to recover)

**Results:** See [Data/clip_results/perturb_length_experiments_baselineseed1_perturbseed0/](Data/clip_results/perturb_length_experiments_baselineseed1_perturbseed0/) (136 training runs)

**Figure 4: Perturbation Recovery Dynamics**

![Recovery Dynamics](Figures/fig4%20(Perturbation%20Recovery)/fig4.png)

#### 5. ViT Baseline Training ([Training/vit_training/baseline/train_vit_sgd.py](Training/vit_training/baseline/train_vit_sgd.py))

Standard supervised training on ImageNet to compare behavioral alignment emergence in classification vs. fine-tuned models:

```python
# Configuration
model = ViT-Base (patch16_224, num_classes=1000)
optimizer = SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR with warmup
epochs = 100
dataset = ImageNet (train/val splits)
```

**Purpose:** Understand how behavioral alignment emerges *naturally* during classification training (without explicit human alignment fine-tuning)

**RSA Measurement:**
- At each epoch, extract ViT embeddings for THINGS images
- Compute RSA against human behavioral RDM
- Track alignment trajectory alongside classification accuracy

**Key Observation:**
- RSA improves gradually: 0.34 (epoch 0) → 0.67+ (epochs 90+)
- Suggests standard supervised learning implicitly learns some human-aligned structure
- But CLIP-HBA fine-tuning achieves higher alignment (0.71) with targeted optimization

**Results:** See [Data/vit_results/rsa_results_final.csv](Data/vit_results/rsa_results_final.csv)

### Demo Instructions

#### Setup

1. **Install Dependencies**
```bash
pip install torch torchvision timm numpy pandas scipy scikit-learn matplotlib seaborn jupyter
pip install open_clip_torch  # For CLIP models
```

2. **Download THINGS Dataset**
```bash
# Visit https://osf.io/jum2f/
# Download THINGS images → Data/Things1854/
# Download CSV files → Data/
# Behavioral RDM (RDM48_triplet.mat) included in repository
```

3. **Verify Data Structure**
Ensure your `Data/` folder contains:
- `clip_results/baseline_clip_results_seed1.csv` (baseline CLIP-HBA results)
- `clip_results/single_sweep_experiments/` (90+ perturbation runs)
- `clip_results/perturb_length_experiments_baselineseed1_perturbseed0/` (length variation experiments)
- `clip_results/image_noise/`, `label_shuffle/`, `uniform_target/`, `target_noise/` (perturbation type comparisons)
- `vit_results/rsa_results_final.csv` (ViT baseline results)
- `vit_results/perturbation_effects.csv` (ViT perturbation analysis)

#### Reproducing Results

**Baseline CLIP-HBA Training:**
```bash
cd Training/clip_behavioral_finetuning/baseline/
python clip_train_behavior_baseline.py
```

**Single-Epoch Perturbation Sweep:**
```bash
cd Training/clip_behavioral_finetuning/uniform_sweep/
python clip_train_behavior_sweep.py \
  --perturb_type random_target \
  --perturb_seed 0 \
  --baseline_seed 1
```

**Variable-Length Perturbation:**
```bash
cd Training/clip_behavioral_finetuning/length_experiments/
python clip_train_behavior_lengths.py \
  --perturb_epoch 10 \
  --perturb_length 20 \
  --perturb_type random_target \
  --output_dir output/e10_l20/ \
  --baseline_dora_directory ../../baseline/output/dora_params/ \
  --baseline_random_state_path ../../baseline/output/random_state_epoch10.pkl \
  --baseline_split_indices_path ../../baseline/output/split_indices.pkl \
  --output_base_directory ./output/
```

**ViT Baseline (requires ImageNet):**
```bash
cd Training/vit_training/baseline/
python train_vit_sgd.py --data_path /path/to/imagenet --output_dir ./checkpoints/
```

#### Analyzing Results

Navigate to `Figures/` and run Jupyter notebooks to reproduce all visualizations:

```bash
cd "Figures/fig1 (Baseline CLIP-HBA Behavioral Alignment)/"
jupyter notebook fig1.ipynb
```

**Figure 1**: Baseline behavioral alignment dynamics
- Plots RSA vs. test loss showing S-shaped learning curve
- Includes ViT baseline comparison (RSA vs. validation loss)
- Demonstrates how alignment evolves during training

**Figure 2**: Effects of different perturbation types
- Compares 4 perturbation types: Image Noise, Blank Image, Label Shuffle, Target Noise
- Shows Δ test loss and Δ behavioral alignment at epochs [5, 15, 25, 35, 45, 70, 98]
- Includes both CLIP and ViT perturbation experiments

**Figure 3**: Single-epoch perturbation sweep
- Tests perturbations applied at each epoch (1-98)
- Shows Δ test loss and Δ behavioral alignment at perturbation epoch
- Reveals early perturbations improve alignment, late perturbations degrade it

**Figure 4**: Perturbation recovery dynamics
- Analyzes how long models take to recover from perturbations of varying lengths
- Shows recovery time (epochs to within 1% of baseline test loss)
- Identifies which perturbations never recover (marked "NR")
- Start epochs tested: [1, 2, 3, 6, 7, 8, 10, 40, 70, 80, 90]
- Lengths tested: [2, 5, 10, 20, 30, 40, 50]

---

## Assessment & Evaluation

### Model Architecture

#### CLIP-HBA-Behavior (Primary Model)

**Architecture:**
- **Backbone**: CLIP-ViT-L/14 (pretrained on 400M image-text pairs)
- **Vision Encoder**: Vision Transformer Large
  - 24 transformer blocks
  - 1024-dimensional embeddings
  - 16 attention heads
  - Patch size: 14×14, Input: 224×224
- **Text Encoder**: Transformer with causal attention
- **Adaptation**: DoRA (Domain-specific Rank Adaptation)
  - Unfrozen layers: Last 2 vision layers + 1 text layer
  - Trainable parameters: ~2.5M (rest frozen)
- **Output**: 66-dimensional behavioral embedding (SPOSEd space)

**Source:** Custom CLIP-HBA implementation based on Zhao et al. (2025)

#### ViT-Base (Baseline Comparison)

**Architecture:**
- **Model**: ViT-Base (patch16_224)
- **Blocks**: 12 transformer encoder layers
- **Embedding Dim**: 768
- **Attention Heads**: 12
- **Parameters**: ~86M
- **Output**: 1000-class ImageNet classification

**Source:** TIMM library ([timm.create_model](https://github.com/huggingface/pytorch-image-models))

### Intended Uses

#### Research Applications

1. **Understanding Human-AI Alignment**
   - Investigating when neural networks develop human-like representations
   - Studying temporal dynamics of behavioral alignment during training
   - Identifying critical periods for cognitive alignment

2. **Robustness Analysis**
   - Testing resilience of representations to training perturbations
   - Understanding recovery dynamics after disruption
   - Probing stability properties of alignment vs. task performance

3. **Neuroscience & Cognitive Modeling**
   - Comparing model representations to human fMRI activity
   - Bridging deep learning and developmental neuroscience
   - Testing critical period hypotheses in artificial systems

4. **Training Methodology Development**
   - Designing temporally-informed training interventions
   - Optimizing fine-tuning strategies for alignment
   - Developing curriculum learning for human-like representations

#### Appropriate Use Cases

- Academic research on vision transformers and human alignment
- Educational demonstrations of training dynamics and critical periods
- Benchmarking representation quality beyond classification accuracy
- Studies on transfer learning from pretrained models to cognitive tasks

#### Inappropriate Uses

- **Production systems**: Models optimized for research, not deployment
- **Real-world applications**: Not validated for safety-critical tasks
- **General-purpose vision**: Specialized for THINGS dataset and behavioral alignment
- **High-stakes decision-making**: Experimental models require human oversight

### Licenses

This project uses the following licensed components:

1. **PyTorch**: BSD-style License
2. **TIMM (PyTorch Image Models)**: Apache License 2.0
3. **CLIP (OpenAI)**: MIT License
4. **OpenCLIP**: Apache License 2.0
5. **THINGS Dataset**: CC BY 4.0 (Creative Commons Attribution)
6. **Natural Object Dataset (NOD)**: CC0 1.0 Universal (fMRI data)

**Project Code License**: MIT License (see end of README)

### Ethical & Bias Considerations

#### Dataset Biases

**THINGS Dataset:**
- Behavioral data collected from **WEIRD populations** (Western, Educated, Industrialized, Rich, Democratic)
- Participants primarily from Western countries—may not generalize to other cultures
- Object categories reflect Western-centric taxonomies
- Triplet judgments represent specific demographic's perceptual similarity, not universal human perception

**ImageNet:**
- Known biases in object representation, geographic diversity, cultural context
- Human images may perpetuate demographic stereotypes
- Class taxonomies reflect Western-centric categorizations

**fMRI Data (NOD):**
- Limited participant pool—neural alignment measured on specific individuals
- Brain activity patterns may vary across demographics, age groups, neurological conditions

#### Model Behavior & Limitations

**Generalization Concerns:**
- High RSA scores indicate alignment with *tested population*, not all humans
- Behavioral alignment is dataset-specific—transfer to other domains unvalidated
- Models learn to align with human biases encoded in similarity judgments

**Perturbation Experiments:**
- Robustness tested under controlled perturbations—does not guarantee adversarial robustness
- Recovery dynamics measured on same distribution (THINGS)—out-of-distribution behavior unknown
- Findings may not generalize to other architectures, datasets, or alignment tasks

**Decoupling of Alignment and Accuracy:**
- Models can achieve high classification accuracy without human-aligned representations
- Conversely, alignment improvements may not correlate with task performance gains
- **Implication**: Standard evaluation metrics (accuracy, loss) are insufficient for assessing cognitive alignment

#### Responsible Use Guidelines

1. **Acknowledge Scope**: Results specific to THINGS dataset and CLIP-HBA model—validate before generalizing
2. **Cite Sources**: Always cite THINGS, NOD, and CLIP papers when using this work
3. **Recognize Bias**: Be explicit about demographic limitations of behavioral and neural data
4. **Avoid Misuse**: Do not use for surveillance, classification of people, or decisions affecting individuals
5. **Human Oversight**: Models are research prototypes—require expert supervision if applied
6. **Reproducibility**: Use provided random seeds, checkpoints, and configurations for replication

#### Broader Impacts

**Positive Impacts:**
- Advances understanding of human-AI alignment mechanisms
- Provides framework for temporally-informed training interventions
- Bridges neuroscience and machine learning through shared evaluation metrics
- Open methodology enables community research on critical periods

**Potential Risks:**
- Findings could be misinterpreted as universal (ignoring demographic specificity)
- Emphasis on "human alignment" might neglect non-Western perspectives
- Perturbation techniques could inform adversarial training methods

**Mitigation:**
- Clear documentation of dataset demographics and limitations
- Encouragement of cross-cultural replication studies
- Emphasis on defensive applications (robustness) over adversarial use

---

## Model & Data Cards

### Model Card: CLIP-HBA-Behavior

**Model Overview**
- **Name**: CLIP with Human Behavioral Alignment (CLIP-HBA-Behavior)
- **Architecture**: CLIP-ViT-L/14 with DoRA adaptation
- **Task**: Predicting 66-dimensional human behavioral embeddings from images
- **Training**: Fine-tuned on THINGS dataset with MSE loss
- **Evaluation**: Representational Similarity Analysis (RSA) against human judgments

**Training Details**
- **Base Model**: OpenAI CLIP-ViT-L/14 (pretrained on 400M image-text pairs)
- **Fine-tuning Data**: 1,806 THINGS images with SPOSEd behavioral embeddings
- **Adaptation Method**: DoRA (last 2 vision layers + 1 text layer unfrozen)
- **Optimizer**: AdamW (lr=3e-4, weight decay)
- **Loss Function**: Mean Squared Error (predicted vs. target embeddings)
- **Training Epochs**: 500 maximum, early stopping patience=20
- **Hardware**: NVIDIA GPUs with distributed training support
- **Training Time**: ~2-4 hours per run (GPU-dependent)

**Evaluation Metrics**
- **Behavioral Alignment**: Spearman ρ(Model RDM, Behavioral RDM) on 48 test images
- **Neural Alignment**: Spearman ρ(Model RDM, fMRI RDM) for ventral visual cortex ROIs
- **Test Loss**: MSE on held-out 48 images
- **Statistical Significance**: p-values computed for all RSA correlations

**Performance**
- **Baseline RSA**: 0.71 (peak behavioral alignment at epoch ~14)
- **Early Perturbation (epoch 1)**: RSA improves to 0.73+
- **Late Perturbation (epoch 90)**: RSA drops to 0.50-0.60
- **Neural Alignment**: Highest in LO2, FFC, PIT (high-level ventral areas)

**Limitations**
- Specialized for THINGS dataset—transfer to other image sets not guaranteed
- Behavioral alignment measured on 48-image test set (small sample)
- Human behavioral data from limited demographic (WEIRD populations)
- Perturbation robustness tested under controlled conditions only

**Intended Use**
- Research on human-AI alignment and critical periods
- Not for production deployment or safety-critical applications
- Educational demonstrations of training dynamics

### Data Card: THINGS Dataset with SPOSEd Embeddings

**Dataset Description**
- **Name**: THINGS (The Human Inference of Natural Geometry and Semantics)
- **Source**: [THINGS Database](https://osf.io/jum2f/) (Hebart et al., 2020)
- **License**: CC BY 4.0
- **Images**: 1,854 naturalistic object images
  - Training: 1,806 images
  - Validation/Test: 48 images
- **Format**: RGB images, various resolutions (resized to 224×224)

**Behavioral Annotations**

**SPOSEd Embeddings (66 Dimensions):**
Derived from 4.7 million human triplet odd-one-out judgments, capturing interpretable semantic dimensions:

- **Material properties**: "metallic; artificial", "textile", "wood-related; brown", "transparent; shiny; crystalline"
- **Category labels**: "food-related", "animal-related", "plant-related", "body; people-related"
- **Functional attributes**: "transportation", "tools-related; handheld", "seating; standing; lying-related"
- **Perceptual features**: "circular; round", "long; thin", "tubular", "spherical; voluminous"
- **Color descriptors**: "white", "black", "red", "green", "yellow", "sand-colored"
- **Texture/pattern**: "coarse-scale pattern; many things", "grid-related; grating-related", "repetitive; spiky"
- **Contextual associations**: "house-related", "bathroom-related", "outdoors", "electronics; technology"

**Full 66D list**: See [Training/functions/spose_dimensions.py](Training/functions/spose_dimensions.py)

**Behavioral RDM (48×48):**
- **File**: [Data/RDM48_triplet.mat](Data/RDM48_triplet.mat)
- **Format**: Representational Dissimilarity Matrix
- **Collection**: Human triplet judgments ("Which is most different: A, B, or C?")
- **Purpose**: Ground-truth benchmark for evaluating model alignment with human perception

**Data Splits**
- **Training**: [Data/spose_embedding66d_rescaled_1806train.csv](Data/spose_embedding66d_rescaled_1806train.csv)
  - 1,806 images with 66D target embeddings
  - Split into 80% train / 20% validation during model training
- **Test**: [Data/spose_embedding66d_rescaled_48val_reordered.csv](Data/spose_embedding66d_rescaled_48val_reordered.csv)
  - 48 held-out images for RSA evaluation
  - Same 48 images used in behavioral RDM

**Data Preprocessing**
- Images resized to 224×224 (ViT/CLIP input size)
- Normalized with CLIP preprocessing: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
- 66D embeddings rescaled to unit norm

**Collection Methodology**
- **Participants**: Online crowdsourcing (predominantly Western participants)
- **Task**: Triplet odd-one-out judgments
- **Quality Control**: Multiple judgments per triplet, outlier removal
- **Dimensionality Reduction**: Sparse Positive Similarity Embedding (SPoSE) algorithm

**Ethical Considerations**
- Images do not contain identifiable human faces
- Behavioral data collected with informed consent
- Object categories reflect Western-centric concepts—cultural bias acknowledged
- Participant demographics not fully reported—limits generalizability

**Citation**
```bibtex
@article{hebart2020revealing,
  title={Revealing the multidimensional mental representations of natural objects underlying human similarity judgments},
  author={Hebart, Martin N and Zheng, Charles Y and Pereira, Francisco and Baker, Chris I},
  journal={Nature Human Behaviour},
  volume={4},
  number={11},
  pages={1173--1185},
  year={2020}
}
```

### Data Card: Natural Object Dataset (fMRI)

**Dataset Description**
- **Name**: Natural Object Dataset (NOD)
- **Source**: [OpenNeuro ds004310](https://openneuro.org/datasets/ds004310) (Gong et al., 2023)
- **License**: CC0 1.0 Universal
- **Modality**: Functional MRI (fMRI) BOLD responses

**fMRI Data Specifications**
- **Participants**: Multi-subject fMRI recordings
- **Stimuli**: 30 naturalistic object images from NOD
- **Regions of Interest (ROIs)**:
  - **Early Visual**: V1, V2, V3 (primary, secondary, tertiary visual cortex)
  - **Ventral Stream**: V4, LO1, LO2, LO3 (lateral occipital areas), FFC (fusiform face complex), PIT (posterior inferotemporal cortex)
- **Preprocessing**: Standard fMRI pipeline (motion correction, spatial smoothing, GLM)

**Neural RDM Construction**
- Extract BOLD responses for 48 THINGS test images (subset of NOD stimuli)
- Compute pairwise dissimilarities between fMRI activation patterns
- Result: 48×48 neural RDM per ROI per participant

**Usage in Project**
- Measure neural alignment: Spearman ρ(Model RDM, Neural RDM)
- Track how perturbations affect alignment with biological visual representations
- Compare behavioral vs. neural alignment dynamics

**Citation**
```bibtex
@article{gong2023large,
  title={A large-scale fMRI dataset for the visual processing of naturalistic scenes},
  author={Gong, Zhengxin and Zhou, Ming and Dai, Yuxuan and Wen, Yushan and Liu, Youyi and Zhen, Zonglei},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={559},
  year={2023}
}
```

---

## Critical Analysis

### Impact of This Project

This research makes several important contributions to understanding human-AI alignment:

#### 1. Critical Periods Exist for Human Alignment Learning

**Discovery**: Behavioral alignment with human perception emerges through **distinct temporal phases** during training, separate from task performance optimization.

- **Early training (epochs 1-10)**: High plasticity for alignment—perturbations can *enhance* behavioral similarity to humans
- **Late training (epochs 60+)**: Low plasticity—perturbations severely degrade alignment, suggesting representations have "locked in"

**Significance**: This parallels biological critical periods in visual cortex development (Hubel & Wiesel, 1970) and extends machine learning critical period research (Achille et al., 2019) to cognitive alignment—a fundamentally new dimension.

#### 2. Alignment and Accuracy Follow Distinct Dynamics

**Discovery**: The temporal structure of learning **differs for human alignment vs. task performance**.

- Epoch 1 perturbations: **Improve alignment** (+0.12 RSA) while **increasing loss**
- Epochs 6-10 perturbations: **Minimal alignment impact** but **large loss increases**
- Late perturbations: **Degrade alignment** but models can still **recover task loss**

**Significance**: This **decoupling** reveals that:
- Standard evaluation metrics (accuracy, loss) do not capture alignment quality
- Models can be accurate without being human-aligned
- Optimizing for task performance ≠ optimizing for cognitive similarity

**Implication**: Future vision systems should be evaluated on **representational geometry**, not just output correctness.

#### 3. Early Perturbations Can Enhance Alignment

**Surprising Finding**: Brief random noise in target embeddings during early training (epochs 1-7) **improves** behavioral and neural alignment.

**Mechanism (Hypothesized):**
- Early perturbations act as implicit **regularizers**, preventing premature convergence to task-specific but non-human-aligned solutions
- Noise increases **representational flexibility**, allowing models to explore broader regions of embedding space
- Similar to beneficial effects of early regularization (Golatkar et al., 2019) and curriculum learning

**Significance**: Challenges the assumption that perturbations always harm learning—**timing determines whether disruption is harmful or helpful**.

#### 4. Recovery Dynamics Reveal Temporal Asymmetries

**Discovery**: Models recover **faster from late perturbations** (5-15 epochs) than early/middle perturbations (20-70 epochs), with some early long perturbations **never recovering**.

**Interpretation**:
- **Late training occupies flatter loss landscapes**—easier to return to baseline trajectory
- **Early training is more fragile**—perturbations can permanently alter developmental path
- Connects to loss landscape research: later training finds broader minima (Zhang et al., 2017)

**Significance**: Fine-tuning strategies should be **stage-dependent**—early interventions have lasting impact, late interventions are more reversible.

### What This Project Reveals

#### Key Insights

**1. When Matters as Much as What**

The **timing** of learning experiences fundamentally shapes the representations models acquire. This has implications for:
- **Curriculum design**: Present data in temporal order that exploits plastic periods
- **Fine-tuning protocols**: Early-stage interventions yield lasting changes; late-stage adjustments are safer but less impactful
- **Data augmentation**: Apply augmentation strategies at stages where they enhance (not harm) target properties

**2. Alignment Requires Explicit Measurement**

Task metrics (loss, accuracy) are **necessary but insufficient** for evaluating human-AI alignment. Our work demonstrates:
- Models can achieve 75% ImageNet accuracy with only 0.67 RSA (ViT baseline)
- CLIP-HBA achieves 0.71 RSA through explicit behavioral optimization
- Perturbations can improve alignment while degrading accuracy (epoch 1)

**Implication**: Representation quality must be assessed through **geometric similarity metrics** (RSA, CKA, etc.), not just downstream task performance.

**3. Neural Networks Exhibit Developmental Structure**

Like biological systems, DNNs show:
- **Phase-like transitions** in learning dynamics (elastic learning window in epochs 5-10)
- **Critical periods** where specific properties (alignment vs. accuracy) are most sensitive
- **Recovery patterns** that depend on developmental stage

**Implication**: Training is not a uniform process—different learning objectives have different sensitive periods.

**4. Robustness is Multidimensional**

Robustness to perturbations is **property-specific**:
- Alignment robustness peaks late in training
- Task loss robustness peaks early in training
- Recovery speed varies by perturbation timing and property measured

**Implication**: Designing robust systems requires understanding what properties need protection and when.

#### Connections to Broader Research

**Mechanistic Interpretability:**
The perturbation framework provides a **causal intervention method** for understanding when and how specific representations form, complementing activation-based interpretability methods.

**AI Safety and Alignment:**
Understanding temporal dynamics of alignment informs:
- When to apply alignment techniques during training
- How to detect misalignment early
- Whether alignment is stable or fragile to continued training

### Next Steps

#### Immediate Extensions

**1. Mechanistic Analysis**

- **Layer-wise RSA**: Compute alignment at each transformer layer to identify where behavioral structure emerges
- **Attention visualization**: Compare attention patterns between high-RSA and low-RSA epochs
- **Dimensionality analysis**: Which of the 66 SPOSEd dimensions are learned first vs. last?

**2. Architectural Comparisons**

- **CNNs vs. ViTs**: Do convolutional architectures show similar critical periods?
- **Model scale**: How do critical periods change with model size (ViT-Small, Base, Large, Huge)?
- **Hybrid models**: Do CNN-ViT hybrids exhibit different temporal dynamics?

**3. Cross-Dataset Generalization**

- **Transfer to other datasets**: Test CLIP-HBA on out-of-distribution images (COCO, ImageNet subsets)
- **Cross-cultural validation**: Collect behavioral data from non-WEIRD populations, measure alignment
- **Other cognitive tasks**: Extend to object recognition time courses, visual search, scene understanding

#### Longer-Term Research Directions

**1. Temporally-Informed Training**

Develop training algorithms that **exploit critical periods**:
- Apply alignment losses during high-plasticity windows (epochs 1-10)
- Apply regularization during solidification phase (epochs 60+)
- Adaptive curricula that adjust based on current alignment level

**2. Multimodal Alignment Dynamics**

Extend to CLIP's full vision-language capabilities:
- Do vision and language alignment emerge at same time or separately?
- Cross-modal perturbations: Disrupt vision, measure language alignment (and vice versa)
- Temporal dynamics of vision-language binding

**3. Continual Learning and Alignment**

- Does alignment persist when models continue training on new tasks?
- Can models learn new alignments without forgetting old ones?
- Critical periods in continual learning scenarios


### Reflections on Methodology

#### Strengths

**Rigorous Experimental Design:**
- 180+ controlled perturbation conditions with systematic parameter sweeps
- Reproducibility through saved random states, checkpoints, split indices
- Multiple evaluation metrics (loss, behavioral RSA, neural RSA)

**Novel Framework:**
- First application of critical period framework to human alignment (not just task performance)
- Perturbation-as-probe methodology provides causal evidence for temporal structure

**Interdisciplinary Bridge:**
- Connects neuroscience (critical periods), psychology (behavioral similarity), and ML (training dynamics)
- Evaluation through both behavioral and neural alignment

#### Limitations

**Computational Cost:**
- 500-epoch runs × 180 conditions = substantial GPU time (weeks of computation)
- Limits exploration of hyperparameter space, additional architectures

**Small Test Set:**
- RSA computed on only 48 images—statistical power limited
- Future work should use larger held-out sets for more robust estimates

**Dataset Specificity:**
- THINGS dataset is relatively small (1,806 training images)
- Object-centric images—may not generalize to scenes, abstract concepts
- WEIRD population bias in behavioral data

**Single Model Family:**
- Focus on CLIP-ViT—unclear if findings generalize to other architectures (CNNs, other ViT variants)
- Future work should test ResNets, EfficientNets, Swin Transformers, etc.

**Perturbation Types:**
- Random target noise is primary perturbation—other perturbation types less explored
- Could test structured noise, adversarial perturbations, data-dependent perturbations

#### Key Lessons

**1. Early Training Matters Immensely**

Small changes in first 10 epochs can have lasting effects—requires careful monitoring and control.

**2. Multiple Metrics Are Essential**

Loss alone would miss alignment improvements from epoch 1 perturbations—geometric evaluation is critical.

**3. Reproducibility Infrastructure is Vital**

Saving random states, split indices, and checkpoints enabled precise replication across 180+ runs.

**4. Visualization Drives Understanding**

Jupyter notebooks for interactive exploration were crucial for discovering temporal patterns in noisy data.

**5. Interdisciplinary Framing Opens New Questions**

Neuroscience-inspired perturbation framework revealed phenomena not visible through standard ML evaluation.

---

## Documentation & Resource Links

### Setup Instructions

#### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for CLIP-ViT-L/14)
- 50GB+ disk space for datasets and checkpoints

#### Installation

**1. Clone Repository**
```bash
git clone <repository-url>
cd ViT-Project
```

**2. Install Python Dependencies**
```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core libraries
pip install timm numpy pandas scipy scikit-learn matplotlib seaborn jupyter

# CLIP support
pip install open_clip_torch

# Additional utilities
pip install tqdm h5py
```

**3. Download THINGS Dataset**
```bash
# Visit https://osf.io/jum2f/
# Download THINGS images → Data/Things1854/
# Download SPOSEd embeddings → Data/
# Behavioral RDM (RDM48_triplet.mat) included in repository
```

**4. Download fMRI Data (Optional)**
```bash
# Visit https://openneuro.org/datasets/ds004310
# Download NOD fMRI data for neural alignment experiments
```

**5. Verify Setup**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import timm; print(f'TIMM: {timm.__version__}')"
python -c "import open_clip; print('OpenCLIP installed')"
```

### Usage Guide

#### Quick Start: Baseline Training

```bash
cd Training/clip_behavioral_finetuning/baseline/
python clip_train_behavior_baseline.py
```

**Expected Output:**
- Training logs: `output/training_YYYYMMDD_HHMMSS.log`
- Results CSV: `output/results_seed1.csv`
- DoRA parameters: `output/dora_params/`
- Random states: `output/random_state_epochX.pkl` (for reproducibility)

**Training time:** ~2-4 hours on single NVIDIA A100

#### Running Single-Epoch Perturbation Sweep

```bash
cd Training/clip_behavioral_finetuning/uniform_sweep/
python clip_train_behavior_sweep.py \
  --perturb_type random_target \
  --perturb_seed 0 \
  --baseline_seed 1 \
  --output_dir output/sweep_random_target/
```

This will run 98 separate training runs (one perturbation per epoch).

#### Running Variable-Length Perturbations

```bash
cd Training/clip_behavioral_finetuning/length_experiments/

# Example: Perturb starting at epoch 10 for 20 epochs
python clip_train_behavior_lengths.py \
  --perturb_epoch 10 \
  --perturb_length 20 \
  --perturb_type random_target \
  --perturb_seed 0 \
  --random_seed 1 \
  --baseline_dora_directory ../../baseline/output/dora_params/ \
  --baseline_random_state_path ../../baseline/output/random_state_epoch10.pkl \
  --baseline_split_indices_path ../../baseline/output/split_indices.pkl \
  --output_dir output/e10_l20/ \
  --output_base_directory ./output/
```

**Batch Processing** (requires SLURM or job scheduler):
```bash
# Example SLURM script for parameter sweep
for epoch in 1 2 3 6 7 8 10 70 80 90; do
  for length in 5 10 20 30 40 50; do
    sbatch run_perturbation.sh --perturb_epoch $epoch --perturb_length $length
  done
done
```

#### Analysis & Visualization

**Load and Plot Baseline Alignment:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load baseline results
baseline = pd.read_csv('Data/clip_results/baseline_clip_results_seed1.csv')

# Plot behavioral alignment vs. test loss (S-curve)
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Test Loss')
ax1.set_ylabel('Behavioral Alignment (RSA)', color='tab:blue')
ax1.plot(baseline['test_loss'], baseline['behavioral_rsa_rho'],
         marker='o', color='tab:blue', label='Behavioral Alignment')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Annotate elastic learning window (epochs 5-10)
elastic_window = baseline[(baseline['epoch'] >= 5) & (baseline['epoch'] <= 10)]
ax1.scatter(elastic_window['test_loss'], elastic_window['behavioral_rsa_rho'],
            s=100, color='orange', label='Elastic Window (epochs 5-10)', zorder=5)

plt.title('CLIP-HBA Behavioral Alignment Dynamics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Compare Perturbation Effects:**
```python
import os
import pandas as pd
import numpy as np
import seaborn as sns

# Load single-epoch sweep results
sweep_dir = 'Data/clip_results/single_sweep_experiments/'
baseline_rsa = baseline.set_index('epoch')['behavioral_rsa_rho']

perturbation_effects = []
for run_dir in os.listdir(sweep_dir):
    csv_path = os.path.join(sweep_dir, run_dir, 'results.csv')
    if os.path.exists(csv_path):
        perturb_df = pd.read_csv(csv_path)
        perturb_epoch = int(run_dir.split('_epoch')[1].split('_')[0])

        # Get RSA at perturbation epoch
        perturb_rsa = perturb_df[perturb_df['epoch'] == perturb_epoch]['behavioral_rsa_rho'].values[0]
        baseline_rsa_val = baseline_rsa.loc[perturb_epoch]

        delta_rsa = perturb_rsa - baseline_rsa_val
        perturbation_effects.append({'epoch': perturb_epoch, 'delta_rsa': delta_rsa})

effects_df = pd.DataFrame(perturbation_effects)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(effects_df['epoch'], effects_df['delta_rsa'],
        color=['green' if x > 0 else 'red' for x in effects_df['delta_rsa']])
plt.xlabel('Perturbation Epoch')
plt.ylabel('Δ Behavioral Alignment')
plt.title('Single-Epoch Perturbation Effects on Behavioral Alignment')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
```

**Heatmap of Length Experiment Results:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load length experiment results
length_dir = 'Data/clip_results/perturb_length_experiments_baselineseed1_perturbseed0/'

results_matrix = []
start_epochs = [1, 2, 3, 6, 7, 8, 10, 70, 80, 90]
lengths = [5, 10, 20, 30, 40, 50]

for start_epoch in start_epochs:
    row = []
    for length in lengths:
        run_name = f'random_target_e{start_epoch}_l{length}'
        csv_path = os.path.join(length_dir, run_name, 'results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            max_rsa = df['behavioral_rsa_rho'].max()
            baseline_max = baseline['behavioral_rsa_rho'].max()
            row.append(max_rsa - baseline_max)
        else:
            row.append(np.nan)
    results_matrix.append(row)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(results_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            xticklabels=lengths, yticklabels=start_epochs,
            cbar_kws={'label': 'Deviation from Baseline'})
plt.xlabel('Perturbation Length')
plt.ylabel('Start Epoch')
plt.title('Maximum Behavioral Alignment Deviations')
plt.tight_layout()
plt.show()
```

### Resource Links

#### Relevant Papers

**1. Vision Transformers & CLIP**

- Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

**2. Human Behavioral Alignment & THINGS Dataset**

- Hebart, M. N., Zheng, C. Y., Pereira, F., & Baker, C. I. (2020). "Revealing the multidimensional mental representations of natural objects underlying human similarity judgments." *Nature Human Behaviour*, 4, 1173-1185. [DOI:10.1038/s41562-020-00951-3](https://doi.org/10.1038/s41562-020-00951-3)

- Zhao, S. C., Hu, Y., Lee, J., Bender, A., Mazumdar, T., Wallace, M., & Tovar, D. A. (2025). "Shifting attention to you: Personalized brain-inspired AI models." *Preprint*.

**3. Critical Periods in Neural Networks**

- Achille, A., Rovere, M., & Soatto, S. (2019). "Critical learning periods in deep networks." *ICLR 2019*. [OpenReview](https://openreview.net/forum?id=BkeStsCcKQ)

- Kleinman, M., Achille, A., & Soatto, S. (2023). "Critical Learning Periods for Multisensory Integration in Deep Networks." *CVPR 2023*. [DOI:10.1109/CVPR52729.2023.02336](https://doi.org/10.1109/CVPR52729.2023.02336)

- Kleinman, M., Achille, A., & Soatto, S. (2024). "Critical learning periods emerge even in deep linear networks." *ICLR 2024*. [OpenReview](https://openreview.net/forum?id=9K8wqqz1aX)

**4. Training Dynamics & Robustness**

- Golatkar, A., Achille, A., & Soatto, S. (2019). "Time matters in regularizing deep networks: Weight decay and data augmentation affect early learning dynamics, matter little near convergence." *NeurIPS 2019*. [arXiv:1905.13277](https://arxiv.org/abs/1905.13277)

- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). "Understanding deep learning requires rethinking generalization." *ICLR 2017*. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)

**5. Representational Similarity Analysis**

- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). "Representational similarity analysis—connecting the branches of systems neuroscience." *Frontiers in Systems Neuroscience*, 2, 4. [DOI:10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)

**6. Developmental Neuroscience (Biological Critical Periods)**

- Hensch, T. K. (2004). "Critical period regulation." *Annual Review of Neuroscience*, 27, 549-579. [DOI:10.1146/annurev.neuro.27.070203.144327](https://doi.org/10.1146/annurev.neuro.27.070203.144327)

- Hubel, D. H., & Wiesel, T. N. (1970). "The period of susceptibility to the physiological effects of unilateral eye closure in kittens." *Journal of Physiology*, 206(2), 419-436. [DOI:10.1113/jphysiol.1970.sp009022](https://doi.org/10.1113/jphysiol.1970.sp009022)

**7. fMRI Dataset**

- Gong, Z., Zhou, M., Dai, Y., Wen, Y., Liu, Y., & Zhen, Z. (2023). "A large-scale fMRI dataset for the visual processing of naturalistic scenes." *Scientific Data*, 10(1), 559. [DOI:10.1038/s41597-023-02471-x](https://doi.org/10.1038/s41597-023-02471-x)

#### Code Bases

1. **CLIP-HBA Official Repository**
   - Repository: [https://github.com/stephenczhao/CLIP-HBA-Official](https://github.com/stephenczhao/CLIP-HBA-Official)
   - Original CLIP-HBA implementation (baseline for this project)

2. **TIMM (PyTorch Image Models)**
   - Repository: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
   - ViT-Base implementation

3. **OpenCLIP**
   - Repository: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
   - CLIP-ViT-L/14 loading and inference

4. **THINGS Database**
   - Website: [https://things-initiative.org/](https://things-initiative.org/)
   - Dataset: [https://osf.io/jum2f/](https://osf.io/jum2f/)

5. **Natural Object Dataset (NOD)**
   - OpenNeuro: [https://openneuro.org/datasets/ds004310](https://openneuro.org/datasets/ds004310)

#### Additional Resources

- **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **Distributed Training (DDP)**: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- **Mixed Precision Training**: [https://pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- **RSA Toolbox (MATLAB/Python)**: [https://github.com/rsagroup/rsatoolbox](https://github.com/rsagroup/rsatoolbox)

---

## Repository Information

### Project Structure Summary

```
ViT-Project/
├── README.md                              # This file
├── Training/                              # All training scripts
│   ├── vit_training/                     # ViT ImageNet baseline
│   ├── clip_behavioral_finetuning/       # CLIP-HBA experiments
│   └── functions/                        # Shared utilities
├── Data/                                 # Datasets and results
│   ├── vit_results/                      # ViT training logs
│   └── clip_results/                     # CLIP-HBA results
└── Figures/                              # Analysis notebooks
    ├── fig1/, fig2/, fig3/, fig4/        # Jupyter notebooks
```


### License

MIT License

Copyright (c) 2025 Seema Dhungana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Acknowledgments

- **THINGS Initiative** for behavioral dataset and SPOSEd embeddings
- **OpenAI** for CLIP pretrained models
- **TIMM Contributors** for ViT implementations
- **Natural Object Dataset** for fMRI data
- **Course Instructors** for guidance on transformer architectures and research methods
- **Achille, Kleinman, & Soatto** for foundational work on critical periods in neural networks

---
