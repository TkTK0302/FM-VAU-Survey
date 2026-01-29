## 📖 Table of contents

- [Our Taxonomy](#taxonomy)
  - [1. Anomaly Perception](#2-Anomaly-Perception)
    - [1.1 Representation Learning](#11-Representation-Learning)
    - [1.2 Vision-Language Alignment](#12-Vision-Language-Alignment)
    - [1.3 Generalization Adaptation](#13-Open-Vocabulary-Generalization)
  - [2. Anomaly Cognition](#2-Anomaly-Cognition)
    - [2.1 Logical Reasoning](#21-Logical-Reasoning)
      - [2.1.1 Explicit Decompositional Reasoning](#211-Explicit-Decompositional-Reasoning)
      - [2.1.2 External Knowledge Expansion](#212-External-Knowledge-Expansion)
      - [2.1.3 Self-Evolving Thinking Process](#213-Self-Evolving-Thinking-Process)
    - [2.2 Assistive Generation](#22-Content-Generation)
      - [2.2.1 Visual Sample Synthesis](#221-Cognitive-Visual-Synthesis)
      - [2.2.2 Holistic Narrative Externalization](#222-Holistic-Narrative-Externalization)
      - [2.2.3 Generative Optimization Guidance](#223-Generative-Optimization-Guidance)
  
## Taxonomy

## 1. Anomaly Perception

### 1.1 Representation Learning

🗓️ **2023**

- 📄 [CLIP-TSA](https://ieeexplore.ieee.org/abstract/document/10222289):Introducing temporal self-attention on CLIP embeddings, 📰 `ICIP` [code](https://github.com/joos2010kj/clip-tsa)

- 📄 [UMIL](https://openaccess.thecvf.com/content/CVPR2023/papers/Lv_Unbiased_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2023_paper.pdf):Mitigating background redundancy through uncertainty estimation, 📰 `CVPR`  [code](https://github.com/ktr-hubrt/UMIL)

🗓️ **2024**

- 📄 [IFS-VAD](https://ieeexplore.ieee.org/document/10720820):Adopting a multi-scale temporal MLP with parallel receptive fields, 📰 `TCSVT` [code](https://github.com/Ria5331/IFS-VAD)

### 1.2 Vision-Language Alignment

🗓️ **2024**

- 📄 [VadCLIP](https://ojs.aaai.org/index.php/AAAI/article/view/28423):Establishing
a vision-language alignment paradigm for anomaly perception by jointly leveraging textual prompts and dual-branch alignment, 📰 `AAAI` [code](https://github.com/nwpu-zxr/VadCLIP) 

- 📄 [TPWNG](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html):Introducing normality-guided prompts and bidirectional semantic constraints, 📰 `CVPR`

- 📄 [ITC](https://ieeexplore.ieee.org/document/10719608):Modeling anomaly-related semantics through learnable prompts and category-aware textual cues, 📰 `TIP`

- 📄 [STPrompt](https://dl.acm.org/doi/abs/10.1145/3664647.3681442):Incorporating spatio-temporal prompts, 📰 `ACM MM`

- 📄 [AnomalyCLIP](https://www.sciencedirect.com/science/article/pii/S1077314224002443):Identifying the normal event subspace and learning text-driven  for abnormal events, 📰 `CVIU` [code](https://github.com/lucazanella/AnomalyCLIP) [homepage](https://lucazanella.github.io/AnomalyCLIP/)

- 📄 [TSTD](https://dl.acm.org/doi/10.1145/3664647.3680934):Explicitly modeling scene semantics to disentangle foreground actions from scene context, 📰 `ACM MM` [code](https://github.com/shengyangsun/TDSD)
  
🗓️ **2025**

- 📄 [PromptVAD](https://ieeexplore.ieee.org/document/11222791):Modeling anomaly-related semantics through learnable prompts and category-aware textual cues, 📰 `TNNLS`

- 📄 [LEC-VAD](https://openreview.net/forum?id=JaNKGPkDpw):Modeling event-level semantic regularities via an anomaly-aware Gaussian mixture, 📰 `ICML`

- 📄 [CMHKF](https://aclanthology.org/2025.acl-long.1524/):Integrating visual, textual, and auditory cues through cross-modal fusion, 📰 `ACL` [code](https://github.com/ssp-seven/CMHKF)

- 📄 [AVadCLIP](https://arxiv.org/abs/2504.04495):Integrating visual, textual, and auditory cues through cross-modal fusion, 📰 `arXiv`

- 📄 [Fed-WSVAD](https://ojs.aaai.org/index.php/AAAI/article/view/35398):Extending this paradigm to federated settings by balancing global semantic consistency and client-specific variability via prompt generation, 📰 `AAAI` [code](https://github.com/wbfwonderful/Fed-WSVAD)

- 📄 [MISSIONGNN](https://openaccess.thecvf.com/content/WACV2025/html/Yun_MissionGNN_Hierarchical_Multimodal_GNN-Based_Weakly_Supervised_Video_Anomaly_Recognition_with_WACV_2025_paper.html):Building mission-specific knowledge graphs with the help of LLMs, 📰 `WACV` [code](https://github.com/c0510gy/MissionGNN)

- 📄 [LocalVAD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html):Aligning semantically relevant local regions with fine-grained textual cues, 📰 `ICLR` [code](https://github.com/AllenYLJiang/Local-Patterns-Generalize-Better/)

🗓️ **2026**

- 📄 [RefineVAD](https://arxiv.org/abs/2511.13204):Injecting soft anomaly category priors into snippet representations via category-prototype-guided refinement, 📰 `AAAI` [code](https://github.com/VisualScienceLab-KHU/RefineVAD)

- 📄 [DSANet](https://arxiv.org/abs/2511.10334):Disentangling normal and abnormal representations to mitigate semantic confusion, 📰 `AAAI` [code](https://github.com/lessiYin/DSANet)

### 1.3 Generalization Adaptation

🗓️ **2024**

- 📄 [OVVAD](https://ieeexplore.ieee.org/document/10654921):Proposing a task decomposition strategy, utilizing semantic knowledge injection and novel anomaly synthesis modules, 📰 `CVPR`

🗓️ **2025**

- 📄 [Anomize](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html):Introducing a multi-level matching mechanism, integrating dynamic action descriptions with static scene features, 📰 `CVPR`
 
- 📄 [PLOVAD](https://ieeexplore.ieee.org/abstract/document/10836858):Devising a dual prompt tuning mechanism that integrates domain-specific learnable vectors with LLM-generated fine-grained semantics, 📰 `TCSVT` [code](https://github.com/ctX-u/PLOVAD)

- 📄 [MEL-OWVAD](https://ieeexplore.ieee.org/abstract/document/10948323):Presenting an evidential deep learning framework, modeling uncertainty via a Dirichlet distribution, 📰 `TMM`

## 2. Anomaly Cognition

### 2.1 Logical Reasoning

#### 2.1.1 Explicit Decompositional Reasoning

🗓️ **2024**

- 📄 [LAVAD](https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.html):Translating visual observations into textual descriptions and performing semantic reasoning over them, 📰 `CVPR` [code](https://github.com/lucazanella/lavad) [homepage](https://lucazanella.github.io/lavad/)

🗓️ **2025**

- 📄 [Unified\_Frame\_VAA](https://openreview.net/pdf?id=Qla5PqFL0s):Organizing anomaly understanding into a unified reasoning pipeline that sequentially performs detection, localization, and explanation, 📰 `NeurIPS` [code](https://github.com/Rathgrith/URF-ZS-HVAA) [homepage](https://rathgrith.github.io/Unified_Frame_VAA/)

- 📄 [EventVAD](https://dl.acm.org/doi/abs/10.1145/3746027.3754500):Decomposing continuous video streams into discrete event units, 📰 `ACM MM` [code](https://github.com/YihuaJerry/EventVAD)

- 📄 [VADTree](https://arxiv.org/abs/2510.22693):Introducing a hierarchical tree structure that supports coarse-to-fine reasoning over anomalous events, 📰 `NeurlPS` [code](https://github.com/wenlongli10/VADTree)

- 📄 [VA-GPT](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html):Introducing structured intermediate reasoning variables, e.g., learnable guiding questions or effective tokens, 📰 `ICCV`

- 📄 [VERA](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.html):Introducing structured intermediate reasoning variables, e.g., learnable guiding questions or effective tokens, 📰 `CVPR` [code](https://github.com/vera-framework/VERA) [homepage](https://vera-framework.github.io/)

- 📄 [HiProbe-VAD](https://dl.acm.org/doi/10.1145/3746027.3755575):A dynamic layer saliency probing mechanism extracting hidden states from optimal intermediate layers for enhanced linear separability of internal anomaly representations, 📰 `ACM MM`

🗓️ **2026**

- 📄 [HeadHunt-VAD](https://arxiv.org/abs/2512.17601):Directly searching for robust anomaly-sensitive attention heads within frozen models to bypass textual decoding, 📰 `AAAI` [code](https://github.com/CebCai/HeadHunt-VAD)

#### 2.1.2 External Knowledge Expansion

🗓️ **2024**

- 📄 [AnomalyRuler](https://arxiv.org/abs/2407.10299):Leveraging induced rules to guide anomaly decisions, enabling models to reason with explicit normality descriptions, 📰 `ECCV` [code](https://github.com/Yuchen413/AnomalyRuler)

🗓️ **2025**

- 📄 [SlowFastVAD](https://arxiv.org/pdf/2504.10320):Leveraging retrieval-augmented generation (RAG) to guide anomaly decisions, enabling models to reason with explicit normality descriptions, 📰 `arXiv` 

- 📄 [MoniTor](https://arxiv.org/abs/2510.21449):Maintaining a memory-based scoring mechanism that anchors current predictions to historical context, 📰 `NeurlPS` [code](https://github.com/YsTvT/MoniTor)

- 📄 [PANDA](https://arxiv.org/abs/2509.26386):Introducing an agentic framework that actively retrieves knowledge, performs self-reflection, and adapts reasoning strategies across scenarios, 📰 `NeurIPS` [code](https://github.com/showlab/PANDA)

- 📄 [HoloTrace](https://dl.acm.org/doi/10.1145/3746027.3755185):Constructing bidirectional causal graphs to trace event evolution and explain anomaly origins, 📰 `ACM MM`

🗓️ **2026**

- 📄 [TargetVAU]:Incorporating explicit spatio-temporal relations to support interpretable anomaly reasoning, 📰 `AAAI`

#### 2.1.3 Self-Evolving Thinking Process

🗓️ **2025**

- 📄 [Vad-R1](https://arxiv.org/abs/2505.19877):Structuring anomaly cognition into multiple complementary aspects, e.g., what, when, where, how, and why, enabling more systematic reasoning refinement, 📰 `NeurlPS` [code](https://github.com/wbfwonderful/Vad-R1)

- 📄 [VAU-R1](https://arxiv.org/abs/2505.23504):Improving reasoning consistency through reward-driven optimization over tasks such as temporal grounding and multi-choice question answering, 📰 `arXiv` [code](https://github.com/GVCLab/VAU-R1) [homepage](https://q1xiangchen.github.io/VAU-R1/)

- 📄 [VAD-DPO](https://openreview.net/pdf?id=crPlJvwHhS):Introducing preference-based learning to correct spurious correlations, guiding models to self-adjust their decision logic, 📰 `NeurlPS`

🗓️ **2026**

- 📄 [CUEBENCH](https://arxiv.org/abs/2511.00613):Pushing self-evolving cognition toward context-aware understanding by modeling conditional anomalies, where anomaly judgments depend on subtle contextual constraints, 📰 `AAAI` [code](https://github.com/Mia-YatingYu/Cue-R1)

### 2.2 Assistive Generation

#### 2.2.1 Visual Sample Synthesis

🗓️ **2025**

- 📄 [PA-VAD](https://arxiv.org/abs/2512.06845):Synthesizing class-specific pseudo-anomalous videos by guiding diffusion models with foundation model-optimized prompts, 📰 `arXiv`

- 📄 [SVTA](https://arxiv.org/abs/2506.01466):Constructing a large-scale synthetic video anomaly retrieval benchmark consisting of over 40,000 video-text pairs across 68 anomaly categories, 📰 `arXiv` [code](https://github.com/Shuyu-XJTU/SVTA)

- 📄 [Pistachio](https://arxiv.org/abs/2511.19474):Focusing on long-form anomaly synthesis, generating scene-conditioned storyline videos with coherent temporal evolution, 📰 `arXiv` [code](https://github.com/Lizruletheworld/Pistachio)

#### 2.2.2 Holistic Narrative Externalization

🗓️ **2024**

- 📄 [CUVA](https://openaccess.thecvf.com/content/CVPR2024/html/Du_Uncovering_What_Why_and_How_A_Comprehensive_Benchmark_for_Causation_CVPR_2024_paper.html):A comprehensive causal understanding benchmark evaluating model capacity to decode what-why-how accident causal chains, 📰 `CVPR` [code](https://github.com/fesvhtr/CUVA)

- 📄 [HAWK](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fca83589e85cb061631b7ebc5db5d6bd-Abstract-Conference.html):Adopting an interactive generative paradigm to structure anomaly understanding via multi-turn dialogues, 📰 `NeurIPS` [code](https://github.com/jqtangust/hawk)

🗓️ **2025**

- 📄 [Ex-VAD](https://openreview.net/pdf?id=xAhUoyb5eU):Converting video frames into textual descriptions and enforcing consistency between narratives and event labels, 📰 `ICML` [code](https://github.com/2004Hrishikesh/Ex-VAD)

🗓️ **2026**

- 📄 [VAGU \& GtS](https://arxiv.org/abs/2507.21507):Organizing generation into a glance-then-scrutinize question-answering process from coarse to fine granularity, 📰 `AAAI`

#### 2.2.3 Generative Optimization Guidance

🗓️ **2025**

- 📄 [Holmes-VAU](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.html):Employing foundation models to generate narrative descriptions at the clip, event, and video levels, which are subsequently inspected and refined by experts to construct hierarchical instruction data, 📰 `CVPR` [code](https://github.com/pipixin321/HolmesVAU)

- 📄 [Vad-R1](https://arxiv.org/abs/2505.19877):Introducing earlier, leverage foundation model-generated texts refined by experts as supervision to fine-tune the policy model, 📰 `NeurlPS` [code](https://github.com/wbfwonderful/Vad-R1)
  

- 📄 [CUEBENCH](https://arxiv.org/abs/2511.00613): refined semantic benchmark fostering high-level conditional reasoning via context-dependent anomaly distinction, 📰 `AAAI` [code]([https://github.com/Mia-YatingYu](https://github.com/Mia-YatingYu/Cue-R1))
