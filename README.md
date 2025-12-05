# Examining the Temporal Dynamics of Human Alignment by Probing Model Learning Windows

## Problem Statement & Overview

Computer vision research has increasingly focused on guiding deep neural networks toward learning more human-like representational structures. However, **little is known about when and how human representational alignment emerges during network learning**. Drawing from developmental neuroscience, where critical periods are identified by observing differential sensitivity to environmental disruptions, we apply **temporally localized perturbations** to probe when human alignment learning is most vulnerable or malleable.

### Key Research Questions

1. **Do critical periods exist for human alignment learning?** Does behavioral alignment with human perception follow distinct temporal dynamics during training, separate from task performance?

2. **When is alignment learning most sensitive to disruption?** Can we identify windows of heightened plasticity or vulnerability for human-like representation learning?

3. **How do models recover from perturbations?** Does perturbation timing and duration affect recovery of human alignment?

### Our Approach

We introduce **perturbations in the target embeddings** of a CLIP behavioral-alignment fine-tuning pipeline (CLIP-HBA-Behavior) and baseline **ViT models trained on ImageNet**, substituting randomly-generated targets for ground-truth behavioral targets at specific epochs and for variable durations.

---

## Methodology

### Models & Training

**CLIP-HBA-Behavior** (Primary Model):
- **Architecture**: CLIP-ViT-L/14 with DoRA adaptation (last 2 vision layers + 1 text layer unfrozen, ~2.5M trainable parameters)
- **Task**: Predict 66D behavioral embeddings (SPOSEd dimensions from THINGS dataset)
- **Training**: AdamW (lr=3e-4), MSE loss, 500 epochs with early stopping
- **Dataset**: 1,806 THINGS images (train/val) + 48 held-out test images
- **Evaluation**: Representational Similarity Analysis (RSA) - Spearman ρ between model RDM and human behavioral RDM

**ViT-Base** (Baseline Comparison):
- **Architecture**: ViT-Base (patch16_224, 86M parameters)
- **Task**: ImageNet classification (1000 classes)
- **Training**: SGD with momentum, cosine annealing, 100 epochs
- **Purpose**: Compare behavioral alignment emergence in classification vs. fine-tuned models

### Perturbation Experiments

**Four perturbation types tested**:
1. **Random Target Noise** (primary): Replace 66D behavioral embeddings with random tensors
2. **Label Shuffling**: Randomly permute target embeddings across samples
3. **Gaussian Image Noise**: Add noise to input images (ε = 0.1)
4. **Blank Images**: Replace inputs with uniform gray

**Experimental paradigms**:
- **Single-Epoch Sweep**: Perturb exactly one epoch at each training stage (epochs 1-98), then resume normal training
- **Variable-Length Perturbations**: Test perturbations at epochs [1, 2, 3, 6, 7, 8, 10, 40, 70, 80, 90] for durations [2, 5, 10, 20, 30, 40, 50] epochs (136 unique conditions)
- **Recovery metric**: First epoch where test loss returns within 1% of baseline

### Connection to Transformers Course

This project applies core transformer concepts to investigate a fundamental scientific question:

- **Vision Transformers & CLIP**: Leveraging pretrained vision-language models for cognitive alignment
- **Parameter-Efficient Fine-Tuning**: DoRA adaptation for efficient behavioral alignment training
- **Training Dynamics**: Temporal analysis reveals phase-like transitions in representation learning

The project bridges **computer vision**, **cognitive neuroscience**, and **developmental learning theory**.

---

## Implementation & Results

### Experimental Results

#### Figure 1: Baseline Behavioral Alignment Trajectories

![CLIP-HBA Baseline](Figures/fig1%20(Baseline%20CLIP-HBA%20Behavioral%20Alignment)/fig1a.png)

**CLIP-HBA Training Dynamics**:
- S-shaped learning curve: RSA improves from 0.46 (epoch 1) → 0.71+ (epochs 10-14)
- **"Elastic learning window"** (epochs 5-10): small loss improvements yield large alignment gains
- Early stopping at epoch ~15 (minimum test loss)

![ViT Baseline](Figures/fig1%20(Baseline%20CLIP-HBA%20Behavioral%20Alignment)/fig1b.png)

**ViT Training Dynamics**:
- Gradual RSA improvement: 0.34 (epoch 0) → 0.67+ (epochs 90+)
- Standard supervised learning implicitly learns some human-aligned structure
- CLIP-HBA achieves higher alignment (0.71) through targeted optimization

#### Figure 2: Effects of Different Perturbation Types

![CLIP Perturbation Types](Figures/fig2%20(Effects%20of%20Different%20Perturbations)/fig2a.png)

**CLIP-HBA**: Target-level disruptions (target noise, label shuffle) are more destructive than input-level (image noise, blank images). Effects increase as training progresses.

![ViT Perturbation Types](Figures/fig2%20(Effects%20of%20Different%20Perturbations)/fig2b.png)

**ViT**: Shows consistent degradation across all perturbation timings, unlike CLIP-HBA's temporal asymmetry.

#### Figure 3: Single-Epoch Perturbation Sweep

![Single-Epoch Sweep Loss](Figures/fig3%20(Single%20Sweep%20Perturbation%20Experiments)/fig3a.png)

![Single-Epoch Sweep Alignment](Figures/fig3%20(Single%20Sweep%20Perturbation%20Experiments)/fig3b.png)

**Key Findings**:

**CLIP-HBA**:
- **Early epochs (1-20)**: Perturbations **improve** alignment (Δ RSA > 0), peak around epochs 5-10
- **Late epochs (60-98)**: Perturbations **degrade** alignment (Δ RSA < 0)
- Loss impact is more uniform across epochs than alignment impact

**Interpretation**: Alignment and accuracy have different critical periods in CLIP-HBA. ViT's uniform degradation indicates standard supervised learning lacks the temporal asymmetry seen in behavioral fine-tuning.

#### Figure 4: Perturbation Recovery Dynamics

![Recovery Dynamics](Figures/fig4%20(Perturbation%20Recovery)/fig4.png)

**Recovery Time Patterns**:
- **Early perturbations (epochs 1-3)**: Moderate recovery (10-60 epochs)
- **Middle perturbations (epochs 6-10)**: Longest recovery times; some **NEVER RECOVER** (marked "NR")
- **Late perturbations (epochs 40-90)**: Fast recovery (5-20 epochs), even for long durations

**Interpretation**: Critical vulnerability window at epochs 6-10 where sustained perturbations cause permanent trajectory alteration. Late training occupies flatter, more stable loss regions.

---

## Assessment & Evaluation

### Intended Uses

**Appropriate**:
- Research on human-AI alignment and critical periods in neural networks
- Benchmarking representation quality beyond classification accuracy
- Educational demonstrations of training dynamics

### Ethical Considerations

**Dataset Biases**:
- THINGS behavioral data from WEIRD populations (Western, Educated, Industrialized, Rich, Democratic)
- May not generalize to other cultures or demographics
- Object categories reflect Western-centric taxonomies

**Model Limitations**:
- High RSA scores indicate alignment with *tested population*, not all humans
- Behavioral alignment is dataset-specific
- Models learn to align with human biases encoded in similarity judgments

### Licenses

- **Code**: MIT License (see end of README)
- **CLIP**: MIT License (OpenAI)
- **TIMM**: Apache License 2.0
- **THINGS Dataset**: CC BY 4.0
- 
---

## Critical Analysis

### Impact of This Project

**1. Critical Periods Exist for Human Alignment Learning**

Behavioral alignment emerges through **distinct temporal phases** separate from task performance:
- Early training (epochs 1-10): High plasticity—perturbations can *enhance* alignment
- Late training (epochs 60+): Low plasticity—perturbations severely degrade alignment

This parallels biological critical periods (Hubel & Wiesel, 1970) and extends ML critical period research to cognitive alignment.

**2. Alignment and Accuracy Follow Distinct Dynamics**

The temporal structure of learning **differs for human alignment vs. task performance**:
- Epoch 1 perturbations: Improve alignment (+0.12 RSA) while increasing loss
- Late perturbations: Degrade alignment but models can still recover task loss
- Standard metrics (accuracy, loss) don't capture alignment quality

**3. Model-Specific Patterns**

CLIP-HBA's early enhancement pattern is **not universal**:
- ViT shows consistent degradation regardless of perturbation timing
- Suggests the reversal pattern is specific to fine-tuning regime
- Different training objectives yield different temporal dynamics

### Next Steps

**Immediate Extensions**:
- Layer-wise RSA to identify where behavioral structure emerges
- Architectural comparisons (CNNs vs. ViTs, model scale effects)
- Cross-dataset generalization and cross-cultural validation

**Longer-Term Directions**:
- Temporally-informed training algorithms that exploit critical periods
- Multimodal alignment dynamics (vision-language binding)
- Continual learning and alignment persistence
- Biological validation with neuroscientists

---

## Documentation & Resources

### Project Structure

```
ViT-Project/
├── Training/                  # Training scripts
│   ├── vit_training/baseline/
│   ├── clip_behavioral_finetuning/
│   │   ├── baseline/
│   │   ├── uniform_sweep/
│   │   └── length_experiments/
│   └── functions/
├── Data/                      # Datasets and results
│   ├── vit_results/
│   └── clip_results/
└── Figures/                   # Analysis notebooks
    ├── fig1/, fig2/, fig3/, fig4/
```

### Key Scripts

**Baseline CLIP-HBA Training**: [Training/clip_behavioral_finetuning/baseline/clip_train_behavior_baseline.py](Training/clip_behavioral_finetuning/baseline/clip_train_behavior_baseline.py)

**Single-Epoch Perturbation Sweep**: [Training/clip_behavioral_finetuning/uniform_sweep/clip_train_behavior_sweep.py](Training/clip_behavioral_finetuning/uniform_sweep/clip_train_behavior_sweep.py)

**Variable-Length Perturbations**: [Training/clip_behavioral_finetuning/length_experiments/clip_train_behavior_lengths.py](Training/clip_behavioral_finetuning/length_experiments/clip_train_behavior_lengths.py)

**ViT Baseline Training**: [Training/vit_training/baseline/train_vit_sgd.py](Training/vit_training/baseline/train_vit_sgd.py)

### Relevant Papers

**Vision Transformers & CLIP**:
- Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

**Human Behavioral Alignment**:
- Hebart et al. (2020). "Revealing the multidimensional mental representations of natural objects." *Nature Human Behaviour*, 4, 1173-1185. [DOI:10.1038/s41562-020-00951-3](https://doi.org/10.1038/s41562-020-00951-3)

**Critical Periods**:
- Achille et al. (2019). "Critical learning periods in deep networks." [ICLR 2019](https://openreview.net/forum?id=BkeStsCcKQ)
- Kleinman et al. (2023). "Critical Learning Periods for Multisensory Integration in Deep Networks." CVPR 2023.

**Training Dynamics**:
- Golatkar et al. (2019). "Time matters in regularizing deep networks." [arXiv:1905.13277](https://arxiv.org/abs/1905.13277)
- Zhang et al. (2017). "Understanding deep learning requires rethinking generalization." [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)

**Representational Similarity**:
- Kriegeskorte et al. (2008). "Representational similarity analysis—connecting the branches of systems neuroscience." *Frontiers in Systems Neuroscience*, 2, 4.

**Developmental Neuroscience**:
- Hubel & Wiesel (1970). "The period of susceptibility to the physiological effects of unilateral eye closure in kittens." *Journal of Physiology*, 206(2), 419-436.

### Code Repositories

- **CLIP-HBA Official**: [github.com/stephenczhao/CLIP-HBA-Official](https://github.com/stephenczhao/CLIP-HBA-Official)
- **TIMM**: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- **OpenCLIP**: [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- **THINGS Database**: [osf.io/jum2f](https://osf.io/jum2f/)
- **Natural Object Dataset**: [openneuro.org/datasets/ds004310](https://openneuro.org/datasets/ds004310)

---

## Appendix

### Setup & Installation

**Prerequisites**:
- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)

**Installation**:
```bash
pip install torch torchvision timm numpy pandas scipy scikit-learn matplotlib seaborn jupyter
pip install open_clip_torch
```

### Running Experiments

**Baseline CLIP-HBA**:
```bash
cd Training/clip_behavioral_finetuning/baseline/
python clip_train_behavior_baseline.py
```

**Single-Epoch Sweep**:
```bash
cd Training/clip_behavioral_finetuning/uniform_sweep/
python clip_train_behavior_sweep.py --perturb_type random_target --perturb_seed 0
```

**Variable-Length Perturbations**:
```bash
cd Training/clip_behavioral_finetuning/length_experiments/
python clip_train_behavior_lengths.py \
  --perturb_epoch 10 --perturb_length 20 \
  --baseline_dora_directory ../../baseline/output/dora_params/ \
  --baseline_random_state_path ../../baseline/output/random_state_epoch10.pkl
```

### Model Cards

**CLIP-HBA-Behavior**:
- **Training Data**: 1,806 THINGS images with SPOSEd 66D embeddings
- **Training Time**: ~2-4 hours per run (GPU-dependent)
- **Performance**: RSA 0.71 (peak behavioral alignment at epoch ~14)
- **Limitations**: Specialized for THINGS dataset, small test set (48 images), WEIRD population bias

**ViT-Base**:
- **Training Data**: ImageNet (1.28M training images, 1000 classes)
- **Performance**: 75% ImageNet accuracy, RSA 0.67 (behavioral alignment)
- **Purpose**: Baseline comparison for alignment emergence during standard supervised learning

### Data Cards

**THINGS Dataset**:
- **Images**: 1,854 naturalistic object images (CC BY 4.0)
- **Behavioral Annotations**: 66D SPOSEd embeddings from 4.7M human triplet judgments
- **Dimensions**: Material properties, category labels, functional attributes, perceptual features, color, texture
- **RDM**: 48×48 representational dissimilarity matrix from human triplet odd-one-out task
- **Full dimension list**: See [Training/functions/spose_dimensions.py](Training/functions/spose_dimensions.py)

**Citation**:
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

---

## License

MIT License

Copyright (c) 2025 Seema Dhungana

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Acknowledgments

- **THINGS Initiative** for behavioral dataset and SPOSEd embeddings
- **OpenAI** for CLIP pretrained models
- **TIMM Contributors** for ViT implementations
- **Course Instructors** for guidance on transformer architectures and research methods
- **Achille, Kleinman, & Soatto** for foundational work on critical periods in neural networks
