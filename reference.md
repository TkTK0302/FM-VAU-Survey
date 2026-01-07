## 📖 Table of contents

- [Our Taxonomy](#taxonomy)
  - [1. Anomaly Perception](#2-Anomaly-Perception)
    - [1.1 Representation Learning](#11-Representation-Learning)
    - [1.2 Vision-Language Alignment](#12-Vision-Language-Alignment)
      - [1.2.1 Semantic Prompt Engineering](#121-Semantic-Prompt-Engineering)
      - [1.2.2 Latent Space Optimization](#122-Latent-Space-Optimization)
    - [1.3 Open-Vocabulary Generalization](#13-Open-Vocabulary-Generalization)
  - [2. Anomaly Cognition](#2-Anomaly-Cognition)
    - [2.1 Logical Reasoning](#21-Logical-Reasoning)
      - [2.1.1 Explicit Decompositional Reasoning](#211-Explicit-Decompositional-Reasoning)
      - [2.1.2 External Knowledge Expansion](#212-External-Knowledge-Expansion)
      - [2.1.3 Self-Evolving Thinking Process](#213-Self-Evolving-Thinking-Process)
      - [2.1.4 Intrinsic Cognitive Probing](#214-Intrinsic-Cognitive-Probing)
    - [2.2 Content Generation](#22-Content-Generation)
      - [2.2.1 Cognitive Visual Synthesis](#221-Cognitive-Visual-Synthesis)
      - [2.2.2 Holistic Narrative Externalization](#222-Holistic-Narrative-Externalization)
      - [2.2.3 Generative Optimization Guidance](#223-Generative-Optimization-Guidance)
  
## Taxonomy

## 1. Anomaly Perception

### 1.1 Representation Learning

🗓️ **2023**

- 📄 [CLIP-TSA](https://ieeexplore.ieee.org/abstract/document/10222289):Leveraging ViT-encoded visual features from CLIP in VAD, 📰 `ICIP` [code](https://github.com/joos2010kj/clip-tsa)

- 📄 [UMIL](https://openaccess.thecvf.com/content/CVPR2023/papers/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.pdf):Leveraging ViT-encoded visual features from CLIP in VAD, 📰 `CVPR` [code](https://github.com/ktr-hubrt/UMIL)

🗓️ **2024**

- 📄 [VadCLIP](https://ojs.aaai.org/index.php/AAAI/article/view/28423):Designing a local-global temporal Adapter to infuse multi-range temporal information into static pre-trained models, 📰 `AAAI` [code](https://github.com/nwpu-zxr/VadCLIP)

- 📄 [IFS-VAD](https://ieeexplore.ieee.org/document/10720820):Integrating parallel multi-scale temporal MLP branches with distinct receptive fields, 📰 `TCSVT` [code](https://github.com/Ria5331/IFS-VAD)

- 📄 [STPrompt](https://dl.acm.org/doi/abs/10.1145/3664647.3681442):Leveraging spatio-temporal prompts to guide visual focus toward localized anomalous regions, 📰 `ACM MM`

### 1.2 Vision-Language Alignment

🗓️ **2024**

- 📄 [VadCLIP](https://ojs.aaai.org/index.php/AAAI/article/view/28423):Establishing vision-language alignment for cross-modal semantic matching, 📰 `AAAI` [code](https://github.com/nwpu-zxr/VadCLIP) 

#### 1.2.1 Semantic Prompt Engineering

🗓️ **2024**

- 📄 [MDFL](https://ieeexplore.ieee.org/document/10657732):Incorporating abnormal-aware prompts for dynamic feature-semantic fusion, 📰 `CVPR` [code](https://github.com/Junxi-Chen/PE-MIL)

- 📄 [TPWNG](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html):Establishing bidirectional semantic constraints via normality guidance for suppressing alignment fluctuations, 📰 `CVPR`

- 📄 [STPrompt](https://dl.acm.org/doi/abs/10.1145/3664647.3681442):A dual-stream spatio-temporal prompt framework for region-level anomaly localization and background suppression, 📰 `ACM MM`

🗓️ **2025**

- 📄 [PromptVAD](https://ieeexplore.ieee.org/document/11222791):Utilizing learnable abnormal prompts via learnable prompts for narrowing semantic gaps, 📰 `TNNLS`

- 📄 [MISSIONGNN](https://openaccess.thecvf.com/content/WACV2025/html/Yun_MissionGNN_Hierarchical_Multimodal_GNN-Based_Weakly_Supervised_Video_Anomaly_Recognition_with_WACV_2025_paper.html):Leveraging Large Language Models (LLMs) to automatically generate mission-specific knowledge graphs and establish multi-point evidence chains, 📰 `WACV` [code](https://github.com/c0510gy/MissionGNN)

- 📄 [Fed-WSVAD](https://ojs.aaai.org/index.php/AAAI/article/view/35398):Fusing global-local features via a text prompt generator for adaptive semantic focus adjustment, 📰 `AAAI` [code](https://github.com/wbfwonderful/Fed-WSVAD)
  
- 📄 [LEC-VAD](https://openreview.net/forum?id=JaNKGPkDpw):A memory-bank prototype learning mechanism to enrich sparse anomaly semantic labels, 📰 `ICML`

#### 1.2.2 Latent Space Optimization

🗓️ **2024**

- 📄 [AnomalyCLIP](https://www.sciencedirect.com/science/article/pii/S1077314224002443):Normal subspace identification for establishing explicit decision boundaries in anomaly detection, 📰 `CVIU` [code](https://lucazanella.github.io/AnomalyCLIP/)
- 📄 [TSTD](https://dl.acm.org/doi/10.1145/3664647.3680934):Explicit foreground-background separation for masking complex background interference, 📰 `ACM MM` [code](https://github.com/shengyangsun/TDSD)

🗓️ **2025**
  
- 📄 [LocalVAD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html):Matching semantic components with textual sub-concepts for resolving critical scene-dependency, 📰 `ICLR` [code](https://github.com/AllenYLJiang/
Local-Patterns-Generalize-Better/)

- 📄 [CMHKF](https://aclanthology.org/2025.acl-long.1524/):Incorporating auditory information for feature compensation to tackle visual occlusions, 📰 `ACL` [code](https://github.com/ssp-seven/CMHKF)
 
- 📄 [AVadCLIP](https://arxiv.org/abs/2504.04495):Developing a collaboration mechanism to enforce temporal correlations for suppressing false alarms, 📰 `arXiv`

🗓️ **2026**

- 📄 [DSANet](https://arxiv.org/abs/2511.10334):A multi-grained disentangled alignment network for explicitly isolating normal and abnormal patterns, 📰 `AAAI` [code](https://github.com/lessiYin/DSANet)

- 📄 [RefineVAD](https://arxiv.org/abs/2511.13204):Motion-aware temporal recalibration for preventing anomalous semantic dilution in global features, 📰 `AAAI`

### 1.3 Open-Vocabulary Generalization

🗓️ **2024**

- 📄 [OVVAD](https://ieeexplore.ieee.org/document/10654921):Implementing task decomposition via knowledge injection and anomaly synthesis to optimize detection for unseen categories, 📰 `CVPR`

🗓️ **2025**

- 📄 [Anomize](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html):Integrating dynamic actions and static scenes via multi-level matching to resolve single-dimensional ambiguity, 📰 `CVPR`
 
- 📄 [PLOVAD](https://ieeexplore.ieee.org/abstract/document/10836858):Devising a dual prompt tuning mechanism that integrates learnable vectors and LLM-generated semantics for adaptive anomaly representation, 📰 `TCSVT` [code](https://github.com/ctX-u/PLOVAD)

- 📄 [MEL-OWVAD](https://ieeexplore.ieee.org/abstract/document/10948323):Modeling uncertainty via a $\text{Dirichlet distribution}$ within an evidential framework to calibrate prediction confidence for out-of-distribution data, 📰 `TMM`

## 2. Anomaly Cognition

### 2.1 Logical Reasoning

#### 2.1.1 Explicit Decompositional Reasoning

🗓️ **2024**

- 📄 [LAVAD](https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.html):Establishing the first training-free framework via visual-to-text translation and prompt engineering to guide semantic reasoning, 📰 `CVPR` [code](https://github.com/lucazanella/lavad) [homepage](https://lucazanella.github.io/lavad/)

- 📄 [AnomalyRuler](https://arxiv.org/abs/2407.10299):Formulating an induction-deduction framework to abstract normal patterns into textual rules for deductive anomaly verification, 📰 `ECCV` [code](https://github.com/Yuchen413/AnomalyRuler)

🗓️ **2025**

- 📄 [Unified\_Frame\_VAA](https://openreview.net/pdf?id=Qla5PqFL0s):Extending the reasoning pipeline into a holistic chained reasoning process, 📰 `NeurIPS` [code](https://github.com/Rathgrith/URF-ZS-HVAA) [homepage](https://rathgrith.github.io/Unified_Frame_VAA/)

- 📄 [EventVAD](https://dl.acm.org/doi/abs/10.1145/3746027.3754500):Proposing an event-aware paradigm to decompose video streams into discrete units, mitigating temporal redundancy from fixed granularity, 📰 `ACM MM` [code](https://github.com/YihuaJerry/EventVAD)

- 📄 [VADTree](https://arxiv.org/abs/2510.22693):Constructing a hierarchical granularity-aware tree to organize the decision space for substantially improved inference efficiency, 📰 `NeurlPS` [code](https://github.com/wenlongli10/VADTree)

- 📄 [VA-GPT](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html):Implementing a fine-grained spatio-temporal token decomposition strategy to focus LLM reasoning on anomaly-salient regions via redundant filtering and temporal prior injection, 📰 `ICCV`

- 📄 [VERA](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.html):Decomposing reasoning into questions for evidence verification to mitigate hallucinations, 📰 `CVPR` [code](https://github.com/vera-framework/VERA) [homepage](https://vera-framework.github.io/)
  
#### 2.1.2 External Knowledge Expansion

🗓️ **2024**

- 📄 [CUVA](https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html):A comprehensive causal understanding benchmark evaluating model capacity to decode what-why-how accident causal chains, 📰 `CVPR` [code](https://github.com/fesvhtr/CUVA)

🗓️ **2025**

- 📄 [SlowFastVAD](https://arxiv.org/pdf/2504.10320):Incorporating a retrieval-augmented mechanism to provide semantic grounding for decision-making via external knowledge retrieval, 📰 `arXiv` 

- 📄 [MoniTor](https://arxiv.org/abs/2510.21449):Introducing a memory-based online scoring queue to anchor current reasoning, 📰 `NeurlPS` [code](https://github.com/YsTvT/MoniTor)

- 📄 [PANDA](https://arxiv.org/abs/2509.26386):An agentic AI engineer endowed with self-adaptive strategy planning capabilities, 📰 `NeurIPS` [code](https://github.com/showlab/PANDA)

- 📄 [HoloTrace](https://dl.acm.org/doi/10.1145/3746027.3755185):Constructing a bidirectional causal knowledge graph to trace root causes for reasoning why anomalies occur beyond mere identification, 📰 `ACM MM`
  
#### 2.1.3 Self-Evolving Thinking Process

🗓️ **2025**

- 📄 [Vad-R1](https://arxiv.org/abs/2505.19877):A perception-to-cognition chain-of-thought optimizing hierarchical reasoning structures via reinforcement learning, 📰 `NeurlPS` [code](https://github.com/wbfwonderful/Vad-R1)

- 📄 [VAU-R1](https://arxiv.org/abs/2505.23504):Introducing reinforcement and supervised fine-tuning strategies via rewarding logically rigorous intermediate reasoning processes, 📰 `arXiv` [code](https://github.com/GVCLab/VAU-R1) [homepage](https://q1xiangchen.github.io/VAU-R1/)

- 📄 [VAD-DPO](https://openreview.net/pdf?id=crPlJvwHhS):Applying direct preference optimization for evidence-aligned reasoning to mitigate statistical shortcuts, 📰 `NeurlPS`

🗓️ **2026**

- 📄 [CUEBENCH](https://arxiv.org/abs/2511.00613):A refined semantic benchmark fostering high-level conditional reasoning via context-dependent anomaly distinction, 📰 `AAAI` [code](https://github.com/Mia-YatingYu/Cue-R1)

#### 2.1.4 Intrinsic Cognitive Probing

🗓️ **2025**

- 📄 [HiProbe-VAD](https://dl.acm.org/doi/10.1145/3746027.3755575):A dynamic layer saliency probing mechanism extracting hidden states from optimal intermediate layers for enhanced linear separability of internal anomaly representations, 📰 `ACM MM`

🗓️ **2026**

- 📄 [HeadHunt-VAD](https://arxiv.org/abs/2512.17601):Directly searching for robust anomaly-sensitive attention heads within frozen models to bypass textual decoding, 📰 `AAAI` [code](https://github.com/CebCai/HeadHunt-VAD)

### 2.2 Content Generation

#### 2.2.1 Cognitive Visual Synthesis

🗓️ **2025**

- 📄 [PA-VAD](https://arxiv.org/abs/2512.06845):Synthesizing semantically controllable pseudo-anomalies via VLM-guided diffusion, 📰 `arXiv`

- 📄 [SVTA](https://arxiv.org/abs/2506.01466):Constructing a synthetic video-text benchmark via generative augmentation covering long-tail anomalies, 📰 `arXiv` [code](https://github.com/Shuyu-XJTU/SVTA)

- 📄 [Pistachio](https://arxiv.org/abs/2511.19474):Developing a scene-conditioned storyline engine synthesizing long-form anomalous videos with coherent event progression, 📰 `arXiv` [code](https://github.com/Lizruletheworld/Pistachio)

#### 2.2.2 Holistic Narrative Externalization

🗓️ **2024**

- 📄 [LAVAD](https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.html):Utilizing foundation models to generate frame-level textual descriptions, 📰 `CVPR` [code](https://github.com/lucazanella/lavad) [homepage](https://lucazanella.github.io/lavad/)

- 📄 [AnomalyRuler](https://arxiv.org/abs/2407.10299):Summarizing normal visual patterns into explicit textual rules, 📰 `ECCV` [code](https://github.com/Yuchen413/AnomalyRuler)

- 📄 [ITC](https://ieeexplore.ieee.org/document/10719608):Generating diverse textual cues across anomaly categories as supplementary semantic context, 📰 `TIP`

- 📄 [HAWK]([https://arxiv.org/abs/2405.16886](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fca83589e85cb061631b7ebc5db5d6bd-Abstract-Conference.html)):Adopting an interactive generative paradigm to structure anomaly understanding via multi-turn dialogues, 📰 `NeurIPS` [code](https://github.com/jqtangust/hawk)

🗓️ **2025**

- 📄 [Ex-VAD](https://openreview.net/pdf?id=xAhUoyb5eU):Enforcing consistency between generated textual descriptions and event labels, 📰 `ICML` [code](https://github.com/2004Hrishikesh/Ex-VAD)

- 📄 [SlowFastVAD](https://arxiv.org/pdf/2504.10320):Leveraging generated textual anomaly definitions as supplementary semantic knowledge, 📰 `arXiv`

🗓️ **2026**

- 📄 [VAGU \& GtS](https://arxiv.org/abs/2507.21507):Organizing generation into a glance-then-scrutinize question-answering process from coarse to fine granularity, 📰 `AAAI`

#### 2.2.3 Generative Optimization Guidance

🗓️ **2025**

- 📄 [Holmes-VAU](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html):Utilizing generated narratives as optimization anchors to align fragmented visual evidence with coherent event-level semantics, 📰 `CVPR` [code](https://github.com/pipixin321/HolmesVAU)

🗓️ **2026**

- 📄 [CUEBENCH](https://arxiv.org/abs/2511.00613): refined semantic benchmark fostering high-level conditional reasoning via context-dependent anomaly distinction, 📰 `AAAI` [code](https://github.com/Mia-YatingYu/Cue-R1)

## Performance Comparison

The following tables are the performance comparison of semi-supervised VAD, weakly supervised VAD, fully supervised VAD, and unsupervised VAD methods as reported in the literature. For semi-supervised, weakly supervised, and unsupervised VAD methods, the evaluation metric used is AUC (%) and AP ( XD-Violence, %), while for  fully supervised VAD methods, the metric is Accuracy (%).

- **Quantitative Performance Comparison of Semi-supervised Methods on Public Datasets.**

| Method     | Publication | Methodology            | Ped1 | Ped2 | Avenue | ShanghaiTech | UBnormal |
| ---------- | -----------:| ---------------------- |:----:|:----:|:------:|:------------:|:--------:|
| AMDN       | BMVC 2015   | One-class classifier   | 92.1 | 90.8 | -      | -            | -        |
| ConvAE     | CVPR 2016   | Reconstruction         | 81.0 | 90.0 | 72.0   | -            | -        |
| STAE       | ACMMM 2017  | Hybrid                 | 92.3 | 91.2 | 80.9   | -            | -        |
| StackRNN   | ICCV 2017   | Sparse coding          | -    | 92.2 | 81.7   | 68.0         | -        |
| FuturePred | CVPR 2018   | Prediction             | 83.1 | 95.4 | 85.1   | 72.8         | -        |
| DeepOC     | TNNLS 2019  | One-class classifier   | 83.5 | 96.9 | 86.6   | -            | -        |
| MemAE      | ICCV 2019   | Reconstruction         | -    | 94.1 | 83.3   | 71.2         | -        |
| AnoPCN     | ACMMM 2019  | Prediction             | -    | 96.8 | 86.2   | 73.6         | -        |
| ObjectAE   | CVPR 2019   | One-class classifier   | -    | 97.8 | 90.4   | 84.9         | -        |
| BMAN       | TIP 2019    | Prediction             | -    | 96.6 | 90.0   | 76.2         | -        |
| sRNN-AE    | TPAMI 2019  | Sparse coding          | -    | 92.2 | 83.5   | 69.6         | -        |
| ClusterAE  | ECCV 2020   | Reconstruction         | -    | 96.5 | 86.0   | 73.3         | -        |
| MNAD       | CVPR 2020   | Reconstruction         | -    | 97.0 | 88.5   | 70.5         | -        |
| VEC        | ACMMM 2020  | Cloze test             | -    | 97.3 | 90.2   | 74.8         | -        |
| AMMC-Net   | AAAI 2021   | Prediction             | -    | 96.6 | 86.6   | 73.7         | -        |
| MPN        | CVPR 2021   | Prediction             | 85.1 | 96.9 | 89.5   | 73.8         | -        |
| HF$^2$-VAD | ICCV 2021   | Hybrid                 | -    | 99.3 | 91.1   | 76.2         | -        |
| BAF        | TPAMI 2021  | One-class classifier   |      | 98.7 | 92.3   | 82.7         | 59.3     |
| Multitask  | CVPR 2021   | Multiple tasks         | -    | 99.8 | 92.8   | 90.2         | -        |
| F$^2$PN    | TPAMI 2022  | Prediction             | 84.3 | 96.2 | 85.7   | 73.0         | -        |
| DLAN-AC    | ECCV 2022   | Reconstruction         | -    | 97.6 | 89.9   | 74.7         | -        |
| BDPN       | AAAI 2022   | Prediction             | -    | 98.3 | 90.3   | 78.1         | -        |
| CAFÉ       | ACMMM 2022  | Prediction             | -    | 98.4 | 92.6   | 77.0         | -        |
| STJP       | ECCV 2022   | Jigsaw puzzle          | -    | 99.0 | 92.2   | 84.3         | 56.4     |
| MPT        | ICCV 2023   | Multiple tasks         | -    | 97.6 | 90.9   | 78.8         | -        |
| HSC        | CVPR 2023   | Hybrid                 | -    | 98.1 | 93.7   | 83.4         | -        |
| LERF       | AAAI 2023   | Predicition            | -    | 99.4 | 91.5   | 78.6         | -        |
| DMAD       | CVPR 2023   | Reconstruction         | -    | 99.7 | 92.8   | 78.8         | -        |
| EVAL       | CVPR 2023   | Interpretable learning | -    | -    | 86.0   | 76.6         | -        |
| FBSC-AE    | CVPR 2023   | Prediction             | -    | -    | 86.8   | 79.2         | -        |
| FPDM       | ICCV 2023   | Prediction             | -    | -    | 90.1   | 78.6         | 62.7     |
| PFMF       | CVPR 2023   | Multiple tasks         | -    | -    | 93.6   | 85.0         | -        |
| STG-NF     | ICCV 2023   | Gaussian classifier    | -    | -    | -      | 85.9         | 71.8     |
| AED-MAE    | CVPR 2024   | Patch inpainting       | -    | 95.4 | 91.3   | 79.1         | 58.5     |
| SSMCTB     | TPAMI 2024  | Patch inpainting       | -    | -    | 91.6   | 83.7         | -        |

- **Quantitative Performance Comparison of Weakly Supervised Methods on Public Datasets.**
  
  | Method   | Publication | Feature        | UCF-Crime | XD-Violence | ShanghaiTech | TAD   |
  | -------- | -----------:| -------------- |:---------:|:-----------:|:------------:|:-----:|
  | DeepMIL  | CVPR 2018   | C3D(RGB)       | 75.40     | -           | -            | -     |
  | GCN      | CVPR 2019   | TSN(RGB)       | 82.12     | -           | 84.44        | -     |
  | HLNet    | ECCV 2020   | I3D(RGB)       | 82.44     | 75.41       | -            | -     |
  | CLAWS    | ECCV 2020   | C3D(RGB)       | 83.03     | -           | 89.67        | -     |
  | MIST     | CVPR 2021   | I3D(RGB)       | 82.30     | -           | 94.83        | -     |
  | RTFM     | ICCV 2021   | I3D(RGB)       | 84.30     | 77.81       | 97.21        | -     |
  | CTR      | TIP 2021    | I3D(RGB)       | 84.89     | 75.90       | 97.48        | -     |
  | MSL      | AAAI 2022   | VideoSwin(RGB) | 85.62     | 78.59       | 97.32        | -     |
  | S3R      | ECCV 2022   | I3D(RGB)       | 85.99     | 80.26       | 97.48        | -     |
  | SSRL     | ECCV 2022   | I3D(RGB)       | 87.43     | -           | 97.98        | -     |
  | CMRL     | CVPR 2023   | I3D(RGB)       | 86.10     | 81.30       | 97.60        | -     |
  | CUPL     | CVPR 2023   | I3D(RGB)       | 86.22     | 81.43       | -            | 91.66 |
  | MGFN     | AAAI 2023   | VideoSwin(RGB) | 86.67     | 80.11       | -            | -     |
  | UMIL     | CVPR 2023   | CLIP           | 86.75     | -           | -            | 92.93 |
  | DMU      | AAAI 2023   | I3D(RGB)       | 86.97     | 81.66       | -            | -     |
  | PE-MIL   | CVPR 2024   | I3D(RGB)       | 86.83     | 88.05       | 98.35        | -     |
  | TPWNG    | CVPR 2024   | CLIP           | 87.79     | 83.68       | -            | -     |
  | VadCLIP  | AAAI 2024   | CLIP           | 88.02     | 84.51       | -            | -     |
  | STPrompt | ACMMM 2024  | CLIP           | 88.08     | -           | 97.81        | -     |

- **Quantitative Performance Comparison of Fully Supervised Methods on Public Datasets.**
  
  | Method       | Publication | Model Input               | Hockey Fights | Violent-Flows | RWF-2000 | Crowed Violence |
  | ------------ | ----------- | ------------------------- | ------------- | ------------- | -------- | --------------- |
  | TS-LSTM      | PR 2016     | RGB+Flow                  | 93.9          | -             | -        | -               |
  | FightNet     | JPCS 2017   | RGB+Flow                  | 97.0          | -             | -        | -               |
  | ConvLSTM     | AVSS 2017   | Frame Difference          | 97.1          | 94.6          | -        | -               |
  | BiConvLSTM   | ECCVW 2018  | Frame Difference          | 98.1          | 96.3          | -        | -               |
  | SPIL         | ECCV 2020   | Skeleton                  | 96.8          | -             | 89.3     | 94.5            |
  | FlowGatedNet | ICPR 2020   | RGB+Flow                  | 98.0          | -             | 87.3     | 88.9            |
  | X3D          | AVSS 2022   | RGB                       | -             | 98.0          | 94.0     | -               |
  | HSCD         | CVIU 2023   | Skeleton+Frame Difference | 94.5          | -             | 90.3     | 94.3            |

- **Quantitative Performance Comparison of Unsupervised Methods on Public Datasets.**
  
  | Method    | Publication | Methodology      | Avenue | Subway Exit | Ped1 | Ped2 | ShaihaiTech | UMN  |
  | --------- | ----------- | ---------------- | ------ | ----------- | ---- | ---- | ----------- | ---- |
  | ADF       | ECCV 2016   | Change detection | 78.3   | 82.4        | -    | -    | -           | 91.0 |
  | Unmasking | ICCV 2017   | Change detection | 80.6   | 86.3        | 68.4 | 82.2 | -           | 95.1 |
  | MC2ST     | BMVC 2018   | Change detection | 84.4   | 93.1        | 71.8 | 87.5 | -           | -    |
  | DAW       | ACMMM 2018  | Pseudo label     | 85.3   | 84.5        | 77.8 | 96.4 | -           | -    |
  | STDOR     | CVPR 2020   | Pseudo label     | -      | 92.7        | 71.7 | 83.2 | -           | 97.4 |
  | TMAE      | ICME 2022   | Change detection | 89.8   | -           | 75.7 | 94.1 | 71.4        | -    |
  | CIL       | AAAI 2022   | Others           | 90.3   | 97.6        | 84.9 | 99.4 | -           | 100  |
  | LBR-SPR   | CVPR 2022   | Others           | 92.8   | -           | 81.1 | 97.2 | 72.6        | -    |
