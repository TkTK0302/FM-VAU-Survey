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

- 📄 [UMIL](https://openaccess.thecvf.com/content/CVPR2023/papers/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.pdf):Leveraging ViT-encoded visual features from CLIP in VAD, 📰 `CVPR`  [code](https://github.com/ktr-hubrt/UMIL)

🗓️ **2024**

- 📄 [VadCLIP](https://ojs.aaai.org/index.php/AAAI/article/view/28423):Designing a local-global temporal Adapter to infuse multi-range temporal information into static pre-trained models, 📰 `AAAI`  [code](https://github.com/nwpu-zxr/VadCLIP)

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
  
- 📄 [LocalVAD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html):Matching semantic components with textual sub-concepts for resolving critical scene-dependency, 📰 `ICLR` [code](https://github.com/AllenYLJiang/Local-Patterns-Generalize-Better/)

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
