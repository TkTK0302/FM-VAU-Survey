## 📖 Table of contents

- [Existing Reviews](#reviews)
- [Our Taxonomy](#taxonomy)
  - [1. Semi-Supervised Video Anomaly Detection](#1-Semi-Supervised-Video-Anomaly-Detection)
    - [1.1 Model Input](#11-Model-Input)
      - [1.1.1 RGB](#111-RGB)
      - [1.1.2 Optical Flow](#112-Optical-Flow)
      - [1.1.3 Skeleton](#113-Skeleton)
      - [1.1.4 Hybrid](#114-Hybrid)
    - [1.2 Methodology](#12-Methodology)
      - [1.2.1 Self-Supervised Learning](#121-Self-Supervised-Learning)
      - [1.2.2 One-Class Learning](#122-One-Class-Learning)
      - [1.2.3 Interpretable Learning](#123-Interpretable-Learning)
    - [1.3 Network Architecture](#13-Network-Architecture)
      - [1.3.1 Auto-Encoder](#131-Auto-Encoder)
      - [1.3.2 GAN](#132-GAN)
      - [1.3.3 Diffusion](#133-Diffusion)
    - [1.4 Model Refinement](#14-Model-Refinement)
      - [1.4.1 Pseudo Anomalies](#141-Pseudo-Anomalies)
      - [1.4.2 Memory Bank](#142-Memory-Bank)
    - [1.5 Model Output](#15-Model-Output)
      - [1.5.1 Frame Level Detection](#151-Frame-Level-Detection)
      - [1.5.2 Pixel Level Detection](#152-Pixel-Level-Detection)
      - [1.5.3 Sentence-Level Description](#153-Sentence-Level-Description)
  - [2. Weakly Supervised Video Anomaly Detection](#2-weakly-supervised-video-anomaly-detection)
    - [2.1 Model Input](#21-Model-Input)
      - [2.1.1 RGB](#211-RGB)
      - [2.1.2 Optical Flow](#212-Optical-Flow)
      - [2.1.3 Audio](#213-Audio)
      - [2.1.4 Text](#214-Text)
      - [2.1.5 Hybrid](#215-Hybrid)
    - [2.2 Methodology](#22-methodology)
      - [2.2.1 One-Stage MIL](#221-One-Stage-MIL)
      - [2.2.2 Two-Stage Self-Training](#222-Two-Stage-Self-Training)
      - [2.2.3 VLM-based Interpretable Learning](#223-VLM-based-Interpretable-Learning)
    - [2.3 Refinement Strategy](#23-Refinement-Strategy)
      - [2.3.1 Temporal Modeling](#231-Temporal-Modeling)
      - [2.3.2 Spatio-Temporal Modeling](#232-Spatio-Temporal-Modeling)
      - [2.3.3 MIL-Based Refinement](#233-MIL-Based-Refinement)
      - [2.3.4 Feature Metric Learning](#234-Feature-Metric-Learning)
      - [2.3.5 Knowledge Distillation](#235-Knowledge-Distillation)
      - [2.3.6 Leveraging Large Models](#236-Leveraging-Large-Models)
    - [2.4 Model Output](#24-Model-Output)
      - [2.4.1 Frame Level Detection](#241-Frame-Level-Detection)
      - [2.4.2 Pixel Level Detection](#242-Pixel-Level-Detection)
      - [2.4.3 Sentence-Level Description](#243-Sentence-Level-Description)
  - [3. Fully Supervised Video Anomaly Detection](#3-Fully-Supervised-Video-Anomaly-Detection)
    - [3.1 Appearance Input](#31-Appearance-Input)
    - [3.2 Motion Input](#32-Motion-Input)
    - [3.3 Skeleton Input](#33-Skeleton-Input)
    - [3.4 Audio Input](#34-Audio-Input)
    - [3.5 Hybrid Input](#35-Hybrid-Input)
  - [4. Unsupervised Video Anomaly Detection](#4-Unsupervised-Video-Anomaly-Detection)
    - [4.1 Pseudo Label Based Paradigm](#41-Pseudo-Label-Based-Paradigm)
    - [4.2 Change Detection Based Paradigm](#42-Change-Detection-Based-Paradigm)
    - [4.3 Others](#43-Others)
  - [5. Open-Set Supervised Video Anomaly Detection](#5-Open-Set-Supervised-Video-Anomaly-Detection)
    - [5.1 Open-Set VAD](#51-Open-Set-VAD)
    - [5.2 Few-Shot VAD](#52-Few-Shot-VAD)
- [Performance Comparison](#performance-comparison)
- [Citation](#citation)

## Reviews

| Reference                                                                           | Year | Venue               | Main Focus                                 | Main Categorization                                              | UVAD | WVAD | SVAD | FVAD | OVAD | LVAD | IVAD |
|:----------------------------------------------------------------------------------- |:----:|:-------------------:|:------------------------------------------:|:----------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|:----:| ---- |
| [Ramachandra et al.](https://ieeexplore.ieee.org/abstract/document/9271895)         | 2020 | IEEE TPAMI          | Semi-supervised single-scene VAD           | Methodology                                                      | ×    | ×    | √    | ×    | ×    | ×    | ×    |
| [Santhosh et al.](https://dl.acm.org/doi/abs/10.1145/3417989)                       | 2020 | ACM CSUR            | VAD applied on road traffic                | Methodology                                                      | √    | ×    | √    | √    | ×    | ×    | ×    |
| [Nayak et al.](https://www.sciencedirect.com/science/article/pii/S0262885620302109) | 2021 | IMAVIS              | Deep learning driven semi-supervised VAD   | Methodology                                                      | ×    | ×    | √    | ×    | ×    | ×    | ×    |
| [Tran et al.](https://dl.acm.org/doi/abs/10.1145/3544014)                           | 2022 | ACM CSUR            | Semi&weakly supervised VAD                 | Architecture                                                     | ×    | ×    | √    | ×    | ×    | ×    | ×    |
| [Chandrakala et al.](https://link.springer.com/article/10.1007/s10462-022-10258-6)  | 2023 | Artif. Intell. Rev. | Deep model-based one&two-class VAD         | Methodology&Architecture                                         | ×    | √    | √    | √    | ×    | ×    | ×    |
| [Liu et al.](https://dl.acm.org/doi/abs/10.1145/3645101)                            | 2023 | ACM CSUR            | Deep models for semi&weakly supervised VAD | Model Input                                                      | √    | √    | √    | √    | ×    | ×    | ×    |
| Our survey                                                                          | 2024 | -                   | Comprehensive VAD taxonomy and deep models | Methodology, Architecture, Refinement, Model Input, Model Output | √    | √    | √    | √    | √    | √    | √    |

*UVAD=Unsupervised VAD, WVAD=Weakly supervised VAD, SVAD=Semi-supervised VAD, FVAD=Fully supervised VAD, OVAD=Open-set supervised VAD, LVAD: Large-model based VAD, IVAD: Interpretable VAD*

## Taxonomy

## 1. Anomaly Perception

### 1.1 Representation Learning

🗓️ **2023**

- 📄 [CLIP-TSA](https://ieeexplore.ieee.org/abstract/document/10222289):Leveraging ViT-encoded visual features from CLIP in VAD, 📰 `ICIP` [code](https://github.com/joos2010kj/clip-tsa) 

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

🗓️ **2026**

- 📄 [DSANet](https://arxiv.org/abs/2511.10334):A multi-grained disentangled alignment network for explicitly isolating normal and abnormal patterns, 📰 `AAAI` [code](https://github.com/lessiYin/DSANet)

- 📄 [RefineVAD](https://arxiv.org/abs/2511.13204):Motion-aware temporal recalibration for preventing anomalous semantic dilution in global features, 📰 `AAAI`
  
#### 1.2.3 Generative-Guided Alignment

🗓️ **2025**
  
- 📄 [LocalVAD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html):Matching semantic components with textual sub-concepts for resolving critical scene-dependency, 📰 `ICLR` [code](https://github.com/AllenYLJiang/
Local-Patterns-Generalize-Better/)

- 📄 [CMHKF](https://aclanthology.org/2025.acl-long.1524/):Incorporating auditory information for feature compensation to tackle visual occlusions, 📰 `ACL` [code](https://github.com/ssp-seven/CMHKF)
 
- 📄 [AVadCLIP](https://arxiv.org/abs/2504.04495):Developing a collaboration mechanism to enforce temporal correlations for suppressing false alarms, 📰 `arXiv`

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

- 📄 []():, 📰 `` [code]()

🗓️ **2025**

- 📄 [Unified\_Frame\_VAA](https://openreview.net/pdf?id=Qla5PqFL0s):Extending the reasoning pipeline into a holistic chained reasoning process, 📰 `NeurIPS` [code](https://github.com/Rathgrith/URF-ZS-HVAA) [homepage](https://rathgrith.github.io/Unified_Frame_VAA/)

- 📄 [EventVAD](https://dl.acm.org/doi/abs/10.1145/3746027.3754500):Proposing an event-aware paradigm to decompose video streams into discrete units, mitigating temporal redundancy from fixed granularity, 📰 `ACM MM` [code](https://github.com/YihuaJerry/EventVAD)

- 📄 [VADTree](https://arxiv.org/abs/2510.22693):Constructing a hierarchical granularity-aware tree to organize the decision space for substantially improved inference efficiency, 📰 `NeurlPS` [code](https://github.com/wenlongli10/VADTree)

- 📄 [VA-GPT](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html):Implementing a fine-grained spatio-temporal token decomposition strategy to focus LLM reasoning on anomaly-salient regions via redundant filtering and temporal prior injection, 📰 `ICCV`

- 📄 [VERA](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.html):Decomposing reasoning into questions for evidence verification to mitigate hallucinations, 📰 `CVPR` [code](https://github.com/vera-framework/VERA) [homepage](https://vera-framework.github.io/)
  
#### 2.1.2 External Knowledge Expansion

🗓️ **2024**

- 📄 []():, 📰 `` [code]()

- 📄 []():, 📰 `` [code]()

- 📄 []():, 📰 `` [code]()

- 📄 []():, 📰 `` [code]()

🗓️ **2025**

- 📄 []():, 📰 `` [code]()

- 📄 []():, 📰 `` [code]()

- 📄 []():, 📰 `` [code]()

- 📄 []():, 📰 `` [code]()
  
#### 2.1.3 Self-Evolving Thinking Process

#### 2.1.4 Intrinsic Cognitive Probing

### 2.2 Content Generation

#### 2.2.1 Cognitive Visual Synthesis

#### 2.2.2 Holistic Narrative Externalization

#### 2.2.3 Generative Optimization Guidance

- 📄 [DeepMIL](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html): Real-world anomaly detectionin surveillance videos, 📰 `CVPR` [code]() [homepage]()


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
