# Awesome AIGC 

<!-- badge link https://github.com/badges/awesome-badges -->
<!-- [![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re) -->
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![issues](https://custom-icon-badges.herokuapp.com/github/issues-raw/weleen/awesome-aigc?logo=issue)](https://github.com/weleen/awesome-aigc/issues "issues")
[![license](https://custom-icon-badges.herokuapp.com/github/license/weleen/awesome-aigc?logo=law&logoColor=white)](https://github.com/weleen/awesome-aigc/blob/main/LICENSE?rgh-link-date=2021-08-09T18%3A10%3A26Z "license MIT")
[![stars](https://custom-icon-badges.herokuapp.com/github/stars/weleen/awesome-aigc?logo=star)](https://github.com/weleen/awesome-aigc/stargazers "stars")

This repo collects papers and repositories about AI-Generated Content . Especially, we focus on the diffusion model, LLM, and MLLM etc.. Please feel free to PR the works missed by the repo. We use the following format to record the papers:
```
[Paper Title](paper link), Conference/Journal, Year, Team | [code](code link), [project](project link)
```


## Table of Contents

- [Awesome AIGC](#awesome-aigc)
  - [Table of Contents](#table-of-contents)
  - [Survey](#survey)
  - [Language](#language)
  - [Vision](#vision)
    - [Image Synthesis](#image-synthesis)
    - [Video Synthesis](#video-synthesis)
    - [3D Synthesis](#3d-synthesis)
    - [Image Editing](#image-editing)
    - [Video Editing](#video-editing)
    - [3D Editing](#3d-editing)
  - [Multimodal](#multimodal)
  - [Others](#others)
  - [Codebase](#codebase)
    - [Image Synthesis](#image-synthesis-1)
    - [Video Synthesis](#video-synthesis-1)
  - [Leadboard](#leadboard)
  - [Links](#links)


## Survey

<details><summary>2024</summary>

- [Efficient Prompting Methods for Large Language Models: A Survey](https://arxiv.org/abs/2404.01077), ArXiv, 2024.
- [Efficient Diffusion Models for Vision: A Survey](https://arxiv.org/abs/2404.01077), ArXiv, 2024.
- [Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward](https://arxiv.org/abs/2402.01799), ArXiv, 2024 | [code](https://github.com/nyunAI/Faster-LLM-Survey)
- [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116), ArXiv, 2024 | [code](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)
- [Model Compression and Efficient Inference for Large Language Models: A Survey](https://arxiv.org/abs/2402.09748), ArXiv, 2024.
- [A Survey on Transformer Compression](https://arxiv.org/abs/2402.05964), ArXiv, 2024.
- [A Comprehensive Survey of Compression Algorithms for Language Models](https://arxiv.org/abs/2401.15347), ArXiv, 2024.
- [Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://arxiv.org/abs/2401.07851), ArXiv, 2024 | [code](https://github.com/hemingkx/Spec-Bench), [project](https://sites.google.com/view/spec-bench)
- [Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security](https://arxiv.org/abs/2401.05459), ArXiv, 2024 | [code](https://github.com/MobileLLM/Personal_LLM_Agents_Survey)
- [A Survey on Hardware Accelerators for Large Language Models](https://arxiv.org/abs/2401.09890), ArXiv, 2024.
- [A Survey of Resource-efficient LLM and Multimodal Foundation Models](https://arxiv.org/abs/2401.08092), ArXiv, 2024 | [code](https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey)
- [Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models](https://arxiv.org/abs/2401.00625), ArXiv, 2024 | [code](https://github.com/tiingweii-shii/Awesome-Resource-Efficient-LLM-Papers)
- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234), ArXiv, 2024.
- [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863), ArXiv, 2024 | [code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)
- [The Efficiency Spectrum of Large Language Models: An Algorithmic Survey](https://arxiv.org/abs/2312.00678), ArXiv, 2024 | [code](https://github.com/tding1/Efficient-LLM-Survey)
- [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633), ArXiv, 2024.
- [A Comprehensive Survey on Knowledge Distillation of Diffusion Models](https://arxiv.org/abs/2304.04262), ArXiv, 2024.
- [Compressing Large-Scale Transformer-Based Models: A Case Study on BERT](https://arxiv.org/abs/2002.11985), TACL, 2024.
- [Understanding LLMs: A Comprehensive Overview from Training to Inference](https://arxiv.org/abs/2401.02038), ArXiv, 2024.
</details>

## Language

## Vision

### Image Synthesis

<details><summary>2024</summary>

- [∞-Brush: Controllable Large Image Synthesis with Diffusion Models in Infinite Dimensions](https://arxiv.org/abs/2407.14709), ECCV, 2024
- [Accelerating Diffusion Sampling with Optimized Time Steps](https://arxiv.org/abs/2402.17376), ECCV, 2024 | [code](https://github.com/scxue/DM-NonUniform)
- [Accelerating Image Generation with Sub-path Linear Approximation Model](https://arxiv.org/abs/2404.13903), ECCV, 2024 | [code](https://github.com/MCG-NJU/SPLAM)
- [AccDiffusion: An Accurate Method for Higher-Resolution Image Generation](https://arxiv.org/abs/2407.10738v1), ECCV, 2024 | [code](https://github.com/lzhxmu/AccDiffusion)
- [AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation](https://arxiv.org/abs/2409.00342), ECCV, 2024 | [code](https://github.com/LeapLabTHU/AdaNAT)
- [AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling](https://arxiv.org/abs/2407.05546v1), ECCV, 2024 | [code](https://github.com/SherryXTChen/AID-Appeal)
- [AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation](https://arxiv.org/abs/2406.18958), ECCV, 2024 | [code](https://github.com/open-mmlab/AnyControl)
- [Arc2Face: A Foundation Model for ID-Consistent Human Faces](https://arxiv.org/abs/2403.11641), ECCV, 2024 | [code](https://github.com/foivospar/Arc2Face)
- [Assessing Sample Quality via the Latent Space of Generative Models](https://arxiv.org/abs/2407.15171), ECCV, 2024 | [code](https://github.com/cvlab-stonybrook/LS-sample-quality)
- [AttentionHand: Text-driven Controllable Hand Image Generation for 3D Hand Reconstruction in the Wild](https://arxiv.org/abs/2407.18034), ECCV, 2024 | [code](https://github.com/redorangeyellowy/AttentionHand)
- [A Watermark-Conditioned Diffusion Model for IP Protection](https://arxiv.org/abs/2403.10893), ECCV, 2024 | [code](https://github.com/rmin2000/WaDiff)
- [Beta-Tuned Timestep Diffusion Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/328_ECCV_2024_paper.php), ECCV, 2024
- [BeyondScene: Higher-Resolution Human-Centric Scene Generation With Pretrained Diffusion](https://arxiv.org/abs/2404.04544), ECCV, 2024 | [code](https://github.com/gwang-kim/BeyondScene)
- [Block-removed Knowledge-distilled Stable Diffusion](https://arxiv.org/abs/2305.15798), ECCV, 2024 | [code](https://github.com/Nota-NetsPresso/BK-SDM)
- [Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation](https://arxiv.org/abs/2403.07860), ECCV, 2024 | [code](https://github.com/ShihaoZhaoZSH/LaVi-Bridge)
- [COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation](https://arxiv.org/abs/2407.11294), ECCV, 2024 | [code](https://github.com/Arking1995/COHO)
- [ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement](https://arxiv.org/abs/2407.07197), ECCV, 2024 | [code](https://github.com/moatifbutt/color-peel)
- [ComFusion: Personalized Subject Generation in Multiple Specific Scenes From Single Image](https://arxiv.org/abs/2402.11849), ECCV, 2024
- [ConceptExpress: Harnessing Diffusion Models for Single-image Unsupervised Concept Extraction](https://arxiv.org/abs/2407.07077), ECCV, 2024 | [code](https://github.com/haoosz/ConceptExpress)
- [ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](https://arxiv.org/abs/2404.07987), ECCV, 2024 | [code](https://github.com/liming-ai/ControlNet_Plus_Plus)
- [Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2037_ECCV_2024_paper.php), ECCV, 2024
- [D4-VTON: Dynamic Semantics Disentangling for Differential Diffusion based Virtual Try-On](https://arxiv.org/abs/2407.15111), ECCV, 2024
- [Data Augmentation for Saliency Prediction via Latent Diffusion](https://arxiv.org/abs/2407.15111), ECCV, 2024 | [code](https://github.com/IVRL/AugSal)
- [DataDream: Few-shot Guided Dataset Generation](https://arxiv.org/abs/2407.10910), ECCV, 2024 | [code](https://github.com/ExplainableML/DataDream)
- [DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation](https://arxiv.org/abs/2409.03755), ECCV, 2024 | [code](https://github.com/wl-zhao/DC-Solver)
- [Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](https://arxiv.org/abs/2310.17316), ECCV, 2024 | [code](https://github.com/EnVision-Research/Defect_Spectrum)
- [Accelerating Diffusion Sampling with Optimized Time Steps](https://arxiv.org/abs/2402.17376), CVPR, 2024 | [code](https://github.com/scxue/DM-NonUniform)
- [Adversarial Score Distillation: When score distillation meets GAN](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_Adversarial_Score_Distillation_When_score_distillation_meets_GAN_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/2y7c3/ASD)
- [Adversarial Text to Continuous Image Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Haydarov_Adversarial_Text_to_Continuous_Image_Generation_CVPR_2024_paper.html), CVPR, 2024
- [Amodal Completion via Progressive Mixed Context Diffusion](https://arxiv.org/abs/2312.15540), CVPR, 2024 | [code](https://github.com/k8xu/amodal)
- [Arbitrary-Scale Image Generation and Upsampling using Latent Diffusion Model and Implicit Neural Decoder](https://arxiv.org/abs/2403.10255), CVPR, 2024 | [code](https://github.com/zhenshij/arbitrary-scale-diffusion)
- [Atlantis: Enabling Underwater Depth Estimation with Stable Diffusion](https://arxiv.org/abs/2312.12471), CVPR, 2024 | [code](https://github.com/zkawfanx/Atlantis)
- [Attention Calibration for Disentangled Text-to-Image Personalization](https://arxiv.org/abs/2403.18551), CVPR, 2024 | [code](https://github.com/Monalissaa/DisenDiff)
- [Attention-Driven Training-Free Efficiency Enhancement of Diffusion Models](https://arxiv.org/abs/2405.05252), CVPR, 2024
- [CapHuman: Capture Your Moments in Parallel Universes](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_CapHuman_Capture_Your_Moments_in_Parallel_Universes_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/VamosC/CapHuman)
- [CHAIN: Enhancing Generalization in Data-Efficient GANs via lipsCHitz continuity constrAIned Normalization](https://arxiv.org/abs/2404.00521), CVPR, 2024
- [Check, Locate, Rectify: A Training-Free Layout Calibration System for Text-to-Image Generation](https://arxiv.org/abs/2311.15773), CVPR, 2024
- [Coarse-to-Fine Latent Diffusion for Pose-Guided Person Image Synthesis](https://arxiv.org/abs/2402.00627), CVPR, 2024 | [code](https://github.com/YanzuoLu/CFLD)
- [CoDi: Conditional Diffusion Distillation for Higher-Fidelity and Faster Image Generation](https://arxiv.org/abs/2310.01407), CVPR, 2024 | [code](https://github.com/fast-codi/CoDi)
- [CoDi-2: In-Context Interleaved and Interactive Any-to-Any Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_CoDi-2_In-Context_Interleaved_and_Interactive_Any-to-Any_Generation_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/microsoft/i-Code/tree/main/CoDi-2)
- [Condition-Aware Neural Network for Controlled Image Generation](https://arxiv.org/abs/2404.01143v1), CVPR, 2024
- [CosmicMan: A Text-to-Image Foundation Model for Humans](https://arxiv.org/abs/2404.01294), CVPR, 2024 | [code](https://github.com/cosmicman-cvpr2024/CosmicMan)
- [Countering Personalized Text-to-Image Generation with Influence Watermarks](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Countering_Personalized_Text-to-Image_Generation_with_Influence_Watermarks_CVPR_2024_paper.html), CVPR, 2024
- [Cross Initialization for Face Personalization of Text-to-Image Models](https://arxiv.org/abs/2312.15905), CVPR, 2024 | [code](https://github.com/lyuPang/CrossInitialization)
- [Customization Assistant for Text-to-image Generation](https://arxiv.org/abs/2312.03045), CVPR, 2024
- [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858), CVPR, 2024 | [code](https://github.com/horseee/DeepCache)
- [DemoFusion: Democratising High-Resolution Image Generation With No $](https://arxiv.org/abs/2311.16973), CVPR, 2024 | [code](https://github.com/PRIS-CV/DemoFusion)
- [Desigen: A Pipeline for Controllable Design Template Generation](https://arxiv.org/abs/2403.09093), CVPR, 2024 | [code](https://github.com/whaohan/desigen)
- [DiffAgent: Fast and Accurate Text-to-Image API Selection with Large Language Model](https://arxiv.org/abs/2404.01342), CVPR, 2024 | [code](https://github.com/OpenGVLab/DiffAgent)
- [Diffusion-driven GAN Inversion for Multi-Modal Face Image Generation](https://arxiv.org/abs/2405.04356v1), CVPR, 2024
- [Diffusion Models Without Attention](https://arxiv.org/abs/2311.18257), CVPR, 2024
- [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/abs/2402.19481), CVPR, 2024 | [code](https://github.com/mit-han-lab/distrifuser)
- [Diversity-aware Channel Pruning for StyleGAN Compression](https://arxiv.org/abs/2403.13548), CVPR, 2024 | [code](https://github.com/jiwoogit/DCP-GAN)
- [Discriminative Probing and Tuning for Text-to-Image Generation](https://www.arxiv.org/abs/2403.04321), CVPR, 2024 | [code](https://github.com/LgQu/DPT-T2I)
- [Domain Gap Embeddings for Generative Dataset Augmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Domain_Gap_Embeddings_for_Generative_Dataset_Augmentation_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/humansensinglab/DoGE)
- [Don’t drop your samples! Coherence-aware training benefits Conditional diffusion](https://arxiv.org/abs/2405.20324), CVPR, 2024 | [code](https://github.com/nicolas-dufour/CAD)
- [Drag Your Noise: Interactive Point-based Editing via Diffusion Semantic Propagation](https://arxiv.org/abs/2404.01050), CVPR, 2024 | [code](https://github.com/haofengl/DragNoise)
- [DREAM: Diffusion Rectification and Estimation-Adaptive Models](https://arxiv.org/abs/2312.00210), CVPR, 2024 | [code](https://github.com/jinxinzhou/DREAM)
- [DreamMatcher: Appearance Matching Self-Attention for Semantically-Consistent Text-to-Image Personalization](https://arxiv.org/abs/2402.09812), CVPR, 2024 | [code](https://github.com/KU-CVLAB/DreamMatcher)
- [Dynamic Prompt Optimizing for Text-to-Image Generation](https://arxiv.org/abs/2404.04095), CVPR, 2024 | [code](https://github.com/Mowenyii/PAE)
- [ECLIPSE: A Resource-Efficient Text-to-Image Prior for Image Generations](https://arxiv.org/abs/2312.04655), CVPR, 2024 | [code](https://github.com/eclipse-t2i/eclipse-inference)
- [Efficient Dataset Distillation via Minimax Diffusion](https://arxiv.org/abs/2311.15529), CVPR, 2024 | [code](https://github.com/vimar-gu/MinimaxDiffusion)
- [ElasticDiffusion: Training-free Arbitrary Size Image Generation](https://arxiv.org/abs/2311.18822), CVPR, 2024 | [code](https://github.com/MoayedHajiAli/ElasticDiffusion-official)
- [EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models](https://arxiv.org/abs/2401.04608), CVPR, 2024 | [code](https://github.com/JingyuanYY/EmoGen)
- [Enabling Multi-Concept Fusion in Text-to-Image Models](https://arxiv.org/abs/2404.03913v1), CVPR, 2024
- [Exact Fusion via Feature Distribution Matching for Few-shot Image Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Exact_Fusion_via_Feature_Distribution_Matching_for_Few-shot_Image_Generation_CVPR_2024_paper.html), CVPR, 2024
- [FaceChain-SuDe: Building Derived Class to Inherit Category Attributes for One-shot Subject-Driven Generation](https://arxiv.org/abs/2403.06775), CVPR, 2024
- [Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094), CVPR, 2024 | [code](https://github.com/zju-pi/diff-sampler)
- [FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition](https://arxiv.org/abs/2312.07536), CVPR, 2024 | [code](https://github.com/genforce/freecontrol)
- [FreeCustom: Tuning-Free Customized Image Generation for Multi-Concept Composition](https://arxiv.org/abs/2405.13870), CVPR, 2024 | [code](https://github.com/aim-uofa/FreeCustom)
- [FreeU: Free Lunch in Diffusion U-Net](https://openaccess.thecvf.com/content/CVPR2024/html/Si_FreeU_Free_Lunch_in_Diffusion_U-Net_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/ChenyangSi/FreeU)
- [Generalizable Tumor Synthesis](https://www.cs.jhu.edu/~alanlab/Pubs24/chen2024towards.pdf), CVPR, 2024 | [code](https://github.com/MrGiovanni/DiffTumor)
- [Generate Like Experts: Multi-Stage Font Generation by Incorporating Font Transfer Process into Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/html/Fu_Generate_Like_Experts_Multi-Stage_Font_Generation_by_Incorporating_Font_Transfer_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/fubinfb/MSD-Font)
- [Generating Daylight-driven Architectural Design via Diffusion Models](https://arxiv.org/abs/2404.13353), CVPR, 2024 | [code](https://github.com/unlimitedli/DDADesign)
- [Generative Unlearning for Any Identity](https://arxiv.org/abs/2405.09879), CVPR, 2024 | [code](https://github.com/JJuOn/GUIDE)
- [HanDiffuser: Text-to-Image Generation With Realistic Hand Appearances](https://arxiv.org/abs/2403.01693), CVPR, 2024 | [code](https://github.com/JJuOn/GUIDE)
- [High-fidelity Person-centric Subject-to-Image Synthesis](https://arxiv.org/abs/2311.10329), CVPR, 2024 | [code](https://github.com/CodeGoat24/Face-diffuser?tab=readme-ov-file)
- [InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization](https://arxiv.org/abs/2404.04650), CVPR, 2024 | [code](https://github.com/xiefan-guo/initno)
- [InstantBooth: Personalized Text-to-Image Generation without Test-Time Finetuning](https://arxiv.org/abs/2304.03411), CVPR, 2024
- [InstanceDiffusion: Instance-level Control for Image Generation](https://arxiv.org/abs/2402.03290), CVPR, 2024 | [code](https://github.com/frank-xwang/InstanceDiffusion)
- [Instruct-Imagen: Image Generation with Multi-modal Instruction](https://arxiv.org/abs/2401.01952), CVPR, 2024
- [Intelligent Grimm - Open-ended Visual Storytelling via Latent Diffusion Models](https://arxiv.org/abs/2306.00973), CVPR, 2024 | [code](https://github.com/haoningwu3639/StoryGen)
- [InteractDiffusion: Interaction-Control for Text-to-Image Diffusion Model](https://arxiv.org/abs/2312.05849), CVPR, 2024 | [code](https://github.com/jiuntian/interactdiffusion)

</details>

### Video Synthesis

<details><summary>2024</summary>

- [Animate Your Motion: Turning Still Images into Dynamic Videos](https://arxiv.org/abs/2403.10179), ECCV, 2024 | [code](https://github.com/Mingxiao-Li/Animate-Your-Motion)
- [Audio-Synchronized Visual Animation](https://arxiv.org/abs/2403.05659), ECCV, 2024 | [code](https://github.com/lzhangbj/ASVA)
- [Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance](https://arxiv.org/abs/2403.14781), ECCV, 2024 | [code](https://github.com/fudan-generative-vision/champ)
- [Dyadic Interaction Modeling for Social Behavior Generation](https://arxiv.org/abs/2403.09069), ECCV, 2024 | [code](https://github.com/Boese0601/Dyadic-Interaction-Modeling)
- [DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors](https://arxiv.org/abs/2310.12190), ECCV, 2024 | [code](https://github.com/Doubiiu/DynamiCrafter)
- [EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](https://arxiv.org/abs/2404.01647), ECCV, 2024 | [code](https://github.com/tanshuai0219/EDTalk)
- [FreeInit: Bridging Initialization Gap in Video Diffusion Models](https://arxiv.org/abs/2312.07537), ECCV, 2024 | [code](https://github.com/TianxingWu/FreeInit)
- [Hybrid Video Diffusion Models with 2D Triplane and 3D Wavelet Representation](https://arxiv.org/abs/2402.13729), ECCV, 2024 | [code](https://github.com/hxngiee/HVDM)
- [IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation](https://arxiv.org/abs/2407.10937), ECCV, 2024 | [code](https://github.com/yhZhai/idol)
- [Kinetic Typography Diffusion Model](https://arxiv.org/abs/2407.10476), ECCV, 2024 | [code](https://github.com/SeonmiP/KineTy)
- [MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2738_ECCV_2024_paper.php), ECCV, 2024
- [MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](https://arxiv.org/abs/2405.20222), ECCV, 2024 | [code](https://github.com/MyNiuuu/MOFA-Video)
- [MotionDirector: Motion Customization of Text-to-Video Diffusion Models](https://arxiv.org/abs/2311.11325), ECCV, 2024
- [MoVideo: Motion-Aware Video Generation with Diffusion Models](https://arxiv.org/abs/2310.08465), ECCV, 2024 | [code](https://github.com/showlab/MotionDirector)
- [Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2790_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/hechang25/MVSD)
- [Noise Calibration: Plug-and-play Content-Preserving Video Enhancement using Pre-trained Video Diffusion Models](https://arxiv.org/abs/2407.10285), ECCV, 2024 | [code](https://github.com/yangqy1110/NC-SDEdit)
- [PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation](https://arxiv.org/abs/2409.18964), ECCV, 2024 | [code](https://github.com/stevenlsw/physgen)
- [VEnhancer: Generative Space-Time Enhancement for Video Generation](https://arxiv.org/abs/2407.15111), ECCV, 2024 | [code](https://github.com/Vchitect/VEnhancer)
- [ZeroI2V: Zero-Cost Adaptation of Pre-trained Transformers from Image to Video](https://arxiv.org/abs/2310.01324), ECCV, 2024 | [code](https://github.com/leexinhao/ZeroI2V)
- [360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model](https://arxiv.org/abs/2401.06578), CVPR, 2024 | [code](https://github.com/Akaneqwq/360DVD)
- [A Recipe for Scaling up Text-to-Video Generation with Text-free Videos](https://arxiv.org/abs/2312.15770), CVPR, 2024 | [code](https://github.com/ali-vilab/VGen)
- [BIVDiff: A Training-Free Framework for General-Purpose Video Synthesis via Bridging Image and Video Diffusion Models](https://arxiv.org/abs/2312.02813), CVPR, 2024 | [code](https://github.com/MCG-NJU/BIVDiff)
- [ConvoFusion: Multi-Modal Conversational Diffusion for Co-Speech Gesture Synthesis](https://arxiv.org/abs/2403.17936), CVPR, 2024 | [code](https://github.com/m-hamza-mughal/convofusion)
- [Co-Speech Gesture Video Generation via Motion-Decoupled Diffusion Model](https://arxiv.org/abs/2404.01862), CVPR, 2024 | [code](https://github.com/thuhcsi/S2G-MDDiffusion)
- [DiffPerformer: Iterative Learning of Consistent Latent Guidance for Diffusion-based Human Video Generation](N/A), CVPR, 2024
- [DisCo: Disentangled Control for Realistic Human Dance Generation](https://arxiv.org/abs/2307.00040), CVPR, 2024 | [code](https://github.com/Wangt-CN/DisCo)
- [FaceChain-ImagineID: Freely Crafting High-Fidelity Diverse Talking Faces from Disentangled Audio](https://arxiv.org/abs/2403.01901), CVPR, 2024
- [Faces that Speak: Jointly Synthesising Talking Face and Speech from Text](https://arxiv.org/abs/2405.10272), CVPR, 2024 | [code](https://github.com/Wangt-CN/DisCo)
- [FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_FlowVid_Taming_Imperfect_Optical_Flows_for_Consistent_Video-to-Video_Synthesis_CVPR_2024_paper.html), CVPR, 2024
- [Generative Rendering: Controllable 4D-Guided Video Generation with 2D Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Generative_Rendering_Controllable_4D-Guided_Video_Generation_with_2D_Diffusion_Models_CVPR_2024_paper.html), CVPR, 2024
- [GenTron: Diffusion Transformers for Image and Video Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GenTron_Diffusion_Transformers_for_Image_and_Video_Generation_CVPR_2024_paper.html), CVPR, 2024
- [Grid Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2404.00234), CVPR, 2024 | [code](https://github.com/taegyeong-lee/Grid-Diffusion-Models-for-Text-to-Video-Generation)
- [Hierarchical Patch-wise Diffusion Models for High-Resolution Video Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Skorokhodov_Hierarchical_Patch_Diffusion_Models_for_High-Resolution_Video_Generation_CVPR_2024_paper.html), CVPR, 2024
- [Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Qing_Hierarchical_Spatio-temporal_Decoupling_for_Text-to-Video_Generation_CVPR_2024_paper.html), CVPR, 2024
- [LAMP: Learn A Motion Pattern for Few-Shot Video Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_LAMP_Learn_A_Motion_Pattern_for_Few-Shot_Video_Generation_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/RQ-Wu/LAMP)
- [Learning Dynamic Tetrahedra for High-Quality Talking Head Synthesis](https://arxiv.org/abs/2402.17364), CVPR, 2024 | [code](https://github.com/zhangzc21/DynTet)
- [Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation guided by the Characteristic Dance Primitives](https://arxiv.org/abs/2403.10518), CVPR, 2024 | [code](https://github.com/li-ronghui/LODGE)
- [MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model](https://arxiv.org/abs/2311.16498), CVPR, 2024 | [code](https://github.com/magic-research/magic-animate)
- [Make-Your-Anchor: A Diffusion-based 2D Avatar Generation Framework](https://arxiv.org/abs/2403.16510), CVPR, 2024 | [code](https://github.com/ICTMCG/Make-Your-Anchor)
- [Make Your Dream A Vlog](https://arxiv.org/abs/2401.09414), CVPR, 2024 | [code](https://github.com/Vchitect/Vlogger)
- [Make Pixels Dance: High-Dynamic Video Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Make_Pixels_Dance_High-Dynamic_Video_Generation_CVPR_2024_paper.html), CVPR, 2024
- [MicroCinema: A Divide-and-Conquer Approach for Text-to-Video Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_MicroCinema_A_Divide-and-Conquer_Approach_for_Text-to-Video_Generation_CVPR_2024_paper.html), CVPR, 2024
- [Panacea: Panoramic and Controllable Video Generation for Autonomous Driving](https://arxiv.org/abs/2311.16813), CVPR, 2024 | [code](https://github.com/wenyuqing/panacea)
- [PEEKABOO: Interactive Video Generation via Masked-Diffusion](https://openaccess.thecvf.com/content/CVPR2024/html/Jain_PEEKABOO_Interactive_Video_Generation_via_Masked-Diffusion_CVPR_2024_paper.html), CVPR, 2024
- [Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners](https://arxiv.org/abs/2308.13712), CVPR, 2024 | [code](https://github.com/yzxing87/Seeing-and-Hearing)
- [SimDA: Simple Diffusion Adapter for Efficient Video Generation](https://arxiv.org/abs/2308.09710), CVPR, 2024 | [code](https://github.com/ChenHsing/SimDA)
- [StyleCineGAN: Landscape Cinemagraph Generation using a Pre-trained StyleGAN](https://arxiv.org/abs/2403.14186), CVPR, 2024 | [code](https://github.com/jeolpyeoni/StyleCineGAN)
- [SyncTalk: The Devil is in the Synchronization for Talking Head Synthesis](https://arxiv.org/abs/2311.17590), CVPR, 2024 | [code](https://github.com/ZiqiaoPeng/SyncTalk)
- [TI2V-Zero: Zero-Shot Image Conditioning for Text-to-Video Diffusion Models](https://arxiv.org/abs/2311.17590), CVPR, 2024 | [code](https://github.com/merlresearch/TI2V-Zero)
- [Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2404.16306), CVPR, 2024 | [code](https://github.com/showlab/Tune-A-Video)
- [VideoBooth: Diffusion-based Video Generation with Image Prompts](https://arxiv.org/abs/2312.00777), CVPR, 2024 | [code](https://github.com/Vchitect/VideoBooth)
- [VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models](https://arxiv.org/abs/2401.09047), CVPR, 2024 | [code](https://github.com/AILab-CVC/VideoCrafter)
- [Video-P2P: Video Editing with Cross-attention Control](https://arxiv.org/abs/2303.04761), CVPR, 2024 | [code](https://github.com/dvlab-research/Video-P2P)

</details>

### 3D Synthesis

<details><summary>2024</summary>

- [BAMM: Bidirectional Autoregressive Motion Model](https://arxiv.org/abs/2403.19435), ECCV, 2024 | [code](https://github.com/exitudio/BAMM)
- [Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation](https://arxiv.org/abs/2407.07554), ECCV, 2024
- [Beyond the Contact: Discovering Comprehensive Affordance for 3D Objects from Pre-trained 2D Diffusion Models](https://arxiv.org/abs/2401.12978), ECCV, 2024 | [code](https://github.com/snuvclab/coma)
- [CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images](https://arxiv.org/abs/2407.04345), ECCV, 2024 | [code](https://github.com/jsshin98/CanonicalFusion)
- [Connecting Consistency Distillation to Score Distillation for Text-to-3D Generation](https://arxiv.org/abs/2407.13584), ECCV, 2024 | [code](https://github.com/LMozart/ECCV2024-GCS-BEG)
- [DiffSurf: A Transformer-based Diffusion Model for Generating and Reconstructing 3D Surfaces in Pose](https://arxiv.org/abs/2408.14860), ECCV, 2024
- [DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1847_ECCV_2024_paper.php), ECCV, 2024
- [DreamDrone: Text-to-Image Diffusion Models are Zero-shot Perpetual View Generators](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2100_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/HyoKong/DreamDrone)
- [DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](https://arxiv.org/abs/2404.06119), ECCV, 2024 | [code](https://github.com/iSEE-Laboratory/DreamView)
- [EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](https://arxiv.org/abs/2405.00915), ECCV, 2024 | [code](https://github.com/ymxlzgy/echoscene)
- [EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/168_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/Frank-ZY-Dou/EMDM)
- [EmoTalk3D: High-Fidelity Free-View Synthesis of Emotional 3D Talking Head](https://arxiv.org/abs/2408.00296), ECCV, 2024
- [Expressive Whole-Body 3D Gaussian Avatar](https://arxiv.org/abs/2408.00297), ECCV, 2024
- [Fast Training of Diffusion Transformer with Extreme Masking for 3D Point Clouds Generation](https://arxiv.org/abs/2312.07231), ECCV, 2024
- [GenerateCT: Text-Conditional Generation of 3D Chest CT Volumes](https://arxiv.org/abs/2405.00915), ECCV, 2024 | [code](https://github.com/ibrahimethemhamamci/GenerateCT)
- [GenRC: Generative 3D Room Completion from Sparse Image Collections](https://arxiv.org/abs/2407.12939), ECCV, 2024 | [code](https://github.com/minfenli/GenRC)
- [GVGEN: Text-to-3D Generation with Volumetric Representation](https://arxiv.org/abs/2403.12957), ECCV, 2024 | [code](https://github.com/SOTAMak1r/GVGEN)
- [Head360: Learning a Parametric 3D Full-Head for Free-View Synthesis in 360°](https://arxiv.org/abs/2407.11174), ECCV, 2024
- [HiFi-123: Towards High-fidelity One Image to 3D Content Generation](https://github.com/AILab-CVC/HiFi-123), ECCV, 2024 | [code](https://arxiv.org/abs/2310.06744)
- [iHuman: Instant Animatable Digital Humans From Monocular Videos](https://arxiv.org/abs/2407.11174), ECCV, 2024
- [JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation](https://arxiv.org/abs/2407.12291), ECCV, 2024
- [KMTalk: Speech-Driven 3D Facial Animation with Key Motion Embedding](https://github.com/ffxzh/KMTalk), ECCV, 2024
- [Length-Aware Motion Synthesis via Latent Diffusion](https://arxiv.org/abs/2407.11532), ECCV, 2024
- [LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/501_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/NIRVANALAN/LN3Diff)
- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](https://arxiv.org/abs/2407.10528), ECCV, 2024
- [MeshAvatar: Learning High-quality Triangular Human Avatars from Multi-view Videos](https://arxiv.org/abs/2407.08414), ECCV, 2024 | [code](https://github.com/shad0wta9/meshavatar)
- [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](https://arxiv.org/abs/2404.19759), ECCV, 2024 | [code](https://github.com/Dai-Wenxun/MotionLCM)
- [Motion Mamba: Efficient and Long Sequence Motion Generation with Hierarchical and Bidirectional Selective SSM](https://arxiv.org/abs/2403.07487), ECCV, 2024 | [code](https://github.com/steve-zeyu-zhang/MotionMamba)
- [MVDiffHD: A Dense High-resolution Multi-view Diffusion Model for Single or Sparse-view 3D Object Reconstruction](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2446_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/Tangshitao/MVDiffusion_plusplus)
- [NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](https://arxiv.org/abs/2403.18241), ECCV, 2024
- [PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidance](https://arxiv.org/abs/2408.02157), ECCV, 2024 | [code](https://github.com/zxcvfd13502/PanoFree)
- [ParCo: Part-Coordinating Text-to-Motion Synthesis](https://arxiv.org/abs/2403.18512), ECCV, 2024 | [code](https://github.com/qrzou/ParCo)
- [Pyramid Diffusion for Fine 3D Large Scene Generation](https://arxiv.org/abs/2311.12085), ECCV, 2024 | [code](https://github.com/yuhengliu02/pyramid-discrete-diffusion)
- [Realistic Human Motion Generation with Cross-Diffusion Models](https://arxiv.org/abs/2312.10993), ECCV, 2024 | [code](https://github.com/THUSIGSICLAB/crossdiff)
- [Repaint123: Fast and High-quality One Image to 3D Generation with Progressive Controllable 2D Repainting](https://arxiv.org/abs/2312.13271), ECCV, 2024 | [code](https://github.com/PKU-YuanGroup/repaint123)
- [RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models](https://arxiv.org/abs/2407.06938), ECCV, 2024 | [code](https://github.com/RodinHD/RodinHD)
- [ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](https://arxiv.org/abs/2407.02040), ECCV, 2024 | [code](https://github.com/theEricMa/ScaleDreamer)
- [ScanTalk: 3D Talking Heads from Unregistered Scans](https://arxiv.org/abs/2403.10942), ECCV, 2024 | [code](https://github.com/miccunifi/ScanTalk)
- [SceneTeller: Language-to-3D Scene Generation](https://arxiv.org/abs/2407.20727), ECCV, 2024
- [StructLDM: Structured Latent Diffusion for 3D Human Generation](https://arxiv.org/abs/2404.01241), ECCV, 2024 | [code](https://github.com/TaoHuUMD/StructLDM)
- [Surf-D: Generating High-Quality Surfaces of Arbitrary Topologies Using Diffusion Models](https://arxiv.org/abs/2311.17050), ECCV, 2024 | [code](https://github.com/Yzmblog/SurfD)
- [SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/150_ECCV_2024_paper.php), ECCV, 2024
- [TexGen: Text-Guided 3D Texture Generation with Multi-view Sampling and Resampling](https://arxiv.org/abs/2408.01291), ECCV, 2024
- [UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/698_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/YG256Li/UniDream)
- [VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](https://arxiv.org/abs/2407.04461), ECCV, 2024
- [VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](https://arxiv.org/abs/2403.12034), ECCV, 2024 | [code](https://github.com/facebookresearch/vfusion3d)
- [Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1890_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/sfanxiang/videoshop)
- [Viewpoint Textual Inversion: Discovering Scene Representations and 3D View Control in 2D Diffusion Models](https://arxiv.org/abs/2309.07986), ECCV, 
- [4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://arxiv.org/abs/2310.08528), CVPR, 2024 | [code](https://github.com/hustvl/4DGaussians)
- [Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling](https://arxiv.org/abs/2311.16096), CVPR, 2024 | [code](https://github.com/lizhe00/AnimatableGaussians)
- [A Unified Approach for Text- and Image-guided 4D Scene Generation](https://arxiv.org/abs/2311.16854), CVPR, 2024 | [code](https://github.com/NVlabs/dream-in-4d)
- [BEHAVIOR Vision Suite: Customizable Dataset Generation via Simulation](https://arxiv.org/abs/2405.09546), CVPR, 2024 | [code](https://github.com/behavior-vision-suite/behavior-vision-suite.github.io)
- [BerfScene: Bev-conditioned Equivariant Radiance Fields for Infinite 3D Scene Generation](https://arxiv.org/abs/2312.02136), CVPR, 2024 | [code](https://github.com/zqh0253/BerfScene)
- [CAD: Photorealistic 3D Generation via Adversarial Distillation](https://arxiv.org/abs/2312.06663), CVPR, 2024 | [code](https://github.com/raywzy/CAD)
- [CAGE: Controllable Articulation GEneration](https://arxiv.org/abs/2312.09570), CVPR, 2024 | [code](https://github.com/3dlg-hcvc/cage)
- [CityDreamer: Compositional Generative Model of Unbounded 3D Cities](https://arxiv.org/abs/2309.00610), CVPR, 2024 | [code](https://github.com/hzxie/CityDreamer)
- [Consistent3D: Towards Consistent High-Fidelity Text-to-3D Generation with Deterministic Sampling Prior](https://arxiv.org/abs/2401.09050), CVPR, 2024 | [code](https://github.com/sail-sg/Consistent3D)
- [ConTex-Human: Free-View Rendering of Human from a Single Image with Texture-Consistent Synthesis](https://arxiv.org/abs/2311.17123), CVPR, 2024 | [code](https://github.com/gaoxiangjun/ConTex-Human)
- [ControlRoom3D: Room Generation using Semantic Proxy Rooms](https://arxiv.org/abs/2312.05208), CVPR, 2024
- [DanceCamera3D: 3D Camera Movement Synthesis with Music and Dance](https://arxiv.org/abs/2403.13667), CVPR, 2024 | [code](https://github.com/Carmenw1203/DanceCamera3D-Official)
- [DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis](https://arxiv.org/abs/2312.13016), CVPR, 2024 | [code](https://github.com/FreedomGu/DiffPortrait3D)
- [DiffSHEG: A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation](https://arxiv.org/abs/2401.04747), CVPR, 2024 | [code](https://github.com/JeremyCJM/DiffSHEG)
- [DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis](https://arxiv.org/abs/2303.14207), CVPR, 2024 | [code](https://github.com/tangjiapeng/DiffuScene)
- [Diffusion 3D Features (Diff3F): Decorating Untextured Shapes with Distilled Semantic Features](https://arxiv.org/abs/2311.17024), CVPR, 2024 | [code](https://github.com/niladridutt/Diffusion-3D-Features)
- [Diffusion Time-step Curriculum for One Image to 3D Generation](https://paperswithcode.com/paper/diffusion-time-step-curriculum-for-one-image), CVPR, 2024 | [code](https://github.com/yxymessi/DTC123)
- [DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models](https://arxiv.org/abs/2304.00916), CVPR, 2024 | [code](https://github.com/yukangcao/DreamAvatar)
- [DreamComposer: Controllable 3D Object Generation via Multi-View Conditions](https://arxiv.org/abs/2312.03611), CVPR, 2024 | [code](https://github.com/yhyang-myron/DreamComposer)
- [DreamControl: Control-Based Text-to-3D Generation with 3D Self-Prior](https://arxiv.org/abs/2312.06439), CVPR, 2024 | [code](https://github.com/tyhuang0428/DreamControl)
- [Emotional Speech-driven 3D Body Animation via Disentangled Latent Diffusion](https://arxiv.org/abs/2312.04466), CVPR, 2024 | [code](https://github.com/kiranchhatre/amuse)
- [EscherNet: A Generative Model for Scalable View Synthesis](https://arxiv.org/abs/2402.03908), CVPR, 2024 | [code](https://github.com/hzxie/city-dreamer)
- [GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models](https://arxiv.org/abs/2310.08529), CVPR, 2024 | [code](https://github.com/hustvl/GaussianDreamer)
- [GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation](https://arxiv.org/abs/2401.04092), CVPR, 2024 | [code](https://github.com/3DTopia/GPTEval3D)
- [Gaussian Shell Maps for Efficient 3D Human Generation](https://arxiv.org/abs/2311.17857), CVPR, 2024 | [code](https://github.com/computational-imaging/GSM)
- [HarmonyView: Harmonizing Consistency and Diversity in One-Image-to-3D](https://arxiv.org/abs/2312.15980), CVPR, 2024 | [code](https://github.com/byeongjun-park/HarmonyView)
- [HIG: Hierarchical Interlacement Graph Approach to Scene Graph Generation in Video Understanding](https://arxiv.org/abs/2312.03050), CVPR, 2024
- [Holodeck: Language Guided Generation of 3D Embodied AI Environments](https://arxiv.org/abs/2312.09067), CVPR, 2024 | [code](https://github.com/allenai/Holodeck)
- [HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation](https://arxiv.org/abs/2310.01406), CVPR, 2024
- [Interactive3D: Create What You Want by Interactive 3D Generation](https://hub.baai.ac.cn/paper/494efc8d-f4ed-4ca4-8469-b882f9489f5e), CVPR, 2024
- [InterHandGen: Two-Hand Interaction Generation via Cascaded Reverse Diffusio](https://arxiv.org/abs/2403.17422), CVPR, 2024 | [code](https://github.com/jyunlee/InterHandGen)
- [Intrinsic Image Diffusion for Single-view Material Estimation](https://arxiv.org/abs/2312.12274), CVPR, 2024 | [code](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion)
- [Make-It-Vivid: Dressing Your Animatable Biped Cartoon Characters from Text](https://arxiv.org/abs/2403.16897), CVPR, 2024 | [code](https://github.com/junshutang/Make-It-Vivid)
- [MoMask: Generative Masked Modeling of 3D Human Motions](https://arxiv.org/abs/2312.00063), CVPR, 2024 | [code](https://github.com/EricGuo5513/momask-codes)
- [MotionEditor: Editing Video Motion via Content-Aware Diffusion](https://openaccess.thecvf.com/content/CVPR2024/html/Tu_MotionEditor_Editing_Video_Motion_via_Content-Aware_Diffusion_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/Francis-Rings/MotionEditor)
- [Editable Scene Simulation for Autonomous Driving via LLM-Agent Collaboration](https://arxiv.org/abs/2402.05746), CVPR, 2024 | [code](https://github.com/yifanlu0227/ChatSim?tab=readme-ov-file)
- [EpiDiff: Enhancing Multi-View Synthesis via Localized Epipolar-Constrained Diffusion](https://arxiv.org/abs/2312.06725), CVPR, 2024 | [code](https://github.com/huanngzh/EpiDiff)
- [OED: Towards One-stage End-to-End Dynamic Scene Graph Generation](https://arxiv.org/abs/2405.16925), CVPR, 2024
- [One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion](https://arxiv.org/abs/2311.07885), CVPR, 2024 | [code](https://github.com/SUDO-AI-3D/One2345plus)
- [Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering](https://arxiv.org/abs/2312.11360), CVPR, 2024 | [code](https://github.com/postech-ami/Paint-it)
- [PEGASUS: Personalized Generative 3D Avatars with Composable Attributes](https://arxiv.org/abs/2402.10636), CVPR, 2024 | [code](https://github.com/snuvclab/pegasus)
- [PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics](https://arxiv.org/abs/2311.12198), CVPR, 2024 | [code](https://github.com/XPandora/PhysGaussian)
- [RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D](https://arxiv.org/abs/2311.16918), CVPR, 2024 | [code](https://github.com/modelscope/richdreamer)
- [SceneTex: High-Quality Texture Synthesis for Indoor Scenes via Diffusion Priors](https://arxiv.org/abs/2311.17261), CVPR, 2024 | [code](https://github.com/daveredrum/SceneTex)
- [SceneWiz3D: Towards Text-guided 3D Scene Composition](https://arxiv.org/abs/2312.08885), CVPR, 2024 | [code](https://github.com/zqh0253/SceneWiz3D)
- [SemCity: Semantic Scene Generation with Triplane Diffusion](https://arxiv.org/abs/2403.07773), CVPR, 2024 | [code](https://github.com/zoomin-lee/SemCity?tab=readme-ov-file)
- [Sherpa3D: Boosting High-Fidelity Text-to-3D Generation via Coarse 3D Prior](https://arxiv.org/abs/2312.06655), CVPR, 2024 | [code](https://github.com/liuff19/Sherpa3D)
- [SIGNeRF: Scene Integrated Generation for Neural Radiance Fields](https://arxiv.org/abs/2401.01647), CVPR, 2024 | [code](https://github.com/cgtuebingen/SIGNeRF)
- [Single Mesh Diffusion Models with Field Latents for Texture Generation](https://arxiv.org/abs/2312.09250), CVPR, 2024 | [code](https://github.com/google-research/google-research/tree/master/mesh_diffusion)
- [SiTH: Single-view Textured Human Reconstruction with Image-Conditioned Diffusion](https://arxiv.org/abs/2311.15855), CVPR, 2024 | [code](https://github.com/SiTH-Diffusion/SiTH)
- [SPAD: Spatially Aware Multiview Diffusers](https://arxiv.org/abs/2402.05235), CVPR, 2024 | [code](https://github.com/yashkant/spad)
- [Text-to-3D Generation with Bidirectional Diffusion using both 2D and 3D priors](https://arxiv.org/abs/2312.04963), CVPR, 2024 | [code](https://github.com/BiDiff/bidiff)
- [Text-to-3D using Gaussian Splatting](https://arxiv.org/abs/2309.16585), CVPR, 2024 | [code](https://github.com/gsgen3d/gsgen)
- [The More You See in 2D, the More You Perceive in 3D](https://arxiv.org/abs/2404.03652), CVPR, 2024 | [code](https://github.com/sap3d/sap3d)
- [Tiger: Time-Varying Denoising Model for 3D Point Cloud Generation with Diffusion Process](https://cvlab.cse.msu.edu/pdfs/Ren_Kim_Liu_Liu_TIGER_supp.pdf), CVPR, 2024 | [code](https://github.com/Zhiyuan-R/Tiger-Diffusion)
- [Towards Realistic Scene Generation with LiDAR Diffusion Models](https://arxiv.org/abs/2404.00815), CVPR, 2024 | [code](https://github.com/hancyran/LiDAR-Diffusion)
- [UDiFF: Generating Conditional Unsigned Distance Fields with Optimal Wavelet Diffusion](https://arxiv.org/abs/2404.06851), CVPR, 2024 | [code](https://github.com/weiqi-zhang/UDiFF)
- [ViVid-1-to-3: Novel View Synthesis with Video Diffusion Models](https://arxiv.org/abs/2312.01305), CVPR, 2024 | [code](https://github.com/ubc-vision/vivid123)

</details>

### Image Editing

<details><summary>2024</summary>

- [A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting](https://arxiv.org/abs/2312.03594), ECCV, 2024 | [code](https://github.com/open-mmlab/PowerPaint)
- [BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion](https://arxiv.org/abs/2403.06976), ECCV, 2024 | [code](https://github.com/TencentARC/BrushNet)
- [COMPOSE: Comprehensive Portrait Shadow Editing](https://arxiv.org/abs/2408.13922), ECCV, 2024
- [CQS: CBAM and Query-Selection Diffusion Model for text-driven Content-aware Image Style Transfer](https://github.com/john09282922/CQS), ECCV, 2024
- [Diffusion-Based Image-to-Image Translation by Noise Correction via Prompt Interpolation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1909_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/JS-Lee525/PIC)
- [DreamSampler: Unifying Diffusion Sampling and Score Distillation for Image Manipulation](https://arxiv.org/abs/2403.11415), ECCV, 2024 | [code](https://github.com/DreamSampler/dream-sampler)
- [EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2096_ECCV_2024_paper.php), ECCV, 2024
- [Efficient 3D-Aware Facial Image Editing via Attribute-Specific Prompt Learning](https://arxiv.org/abs/2406.04413), ECCV, 2024 | [code](https://github.com/VIROBO-15/Efficient-3D-Aware-Facial-Image-Editing)
- [Enhanced Controllability of Diffusion Models via Feature Disentanglement and Realism-Enhanced Sampling Methods](https://arxiv.org/abs/2302.14368), ECCV, 2024
- [Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2157_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/furiosa-ai/eta-inversion)
- [Every Pixel Has its Moments: Ultra-High-Resolution Unpaired Image-to-Image Translation via Dense Normalization](https://arxiv.org/abs/2407.04245), ECCV, 2024 | [code](https://github.com/Kaminyou/Dense-Normalization)
- [Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control](https://arxiv.org/abs/2407.04245), ECCV, 2024 | [code](https://github.com/FaceAdapter/Face-Adapter)
- [Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation](https://arxiv.org/abs/2312.14223), ECCV, 2024 | [code](https://github.com/nina-weng/FastDiME_Med)
- [FlexiEdit: Frequency-Aware Latent Refinement for Enhanced Non-Rigid Editing](https://arxiv.org/abs/2405.12970), ECCV, 2024 | [code](https://github.com/kookie12/FlexiEdit)
- [FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/759_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/Thermal-Dynamics/FreeDiff)
- [GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections](https://arxiv.org/abs/2409.12952), ECCV, 2024 | [code](https://github.com/trustinai/gdvaecode)
- [Guide-and-Rescale: Self-Guidance Mechanism for Effective Tuning-Free Real Image Editing](https://arxiv.org/abs/2409.01322), ECCV, 2024 | [code](https://github.com/FusionBrainLab/Guide-and-Rescale)
- [GroupDiff: Diffusion-based Group Portrait Editing](https://arxiv.org/abs/2409.14379), ECCV, 2024 | [code](https://github.com/yumingj/GroupDiff)
- [InstaStyle: Inversion Noise of a Stylized Image is Secretly a Style Adviser](https://arxiv.org/abs/2311.15040), ECCV, 2024 | [code](https://github.com/cuixing100876/InstaStyle)
- [InstructGIE: Towards Generalizable Image Editing](https://arxiv.org/abs/2403.05018), ECCV, 2024
- [Lazy Diffusion Transformer for Interactive Image Editing](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3436_ECCV_2024_paper.php), ECCV, 2024
- [Leveraging Text Localization for Scene Text Removal via Text-aware Masked Image Modeling](https://arxiv.org/abs/2409.13431), ECCV, 2024 | [code](https://github.com/wzx99/TMIM)
- [MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](https://arxiv.org/abs/2409.00674), ECCV, 2024
- [Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](https://arxiv.org/abs/2308.14469), ECCV, 2024 | [code](https://github.com/yangxy/PASD)
- [RadEdit: stress-testing biomedical vision models via diffusion image editing](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1923_ECCV_2024_paper.php), ECCV, 2024
- [Real-time 3D-aware Portrait Editing from a Single Image](https://arxiv.org/abs/2402.14000), ECCV, 2024 | [code](https://github.com/EzioBy/3dpe)
- [RegionDrag: Fast Region-Based Image Editing with Diffusion Models](https://arxiv.org/abs/2407.18247), ECCV, 2024 | [code](https://github.com/Visual-AI/RegionDrag)
- [Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](https://arxiv.org/abs/2402.12688), ECCV, 2024 | [code](https://github.com/hurunyi/Robust-Wide)
- [ShoeModel: Learning to Wear on the User-specified Shoes via Diffusion Model](https://arxiv.org/abs/2404.04833), ECCV, 2024
- [Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](https://arxiv.org/abs/2403.11105), ECCV, 2024 | [code](https://github.com/leeruibin/SPDInv)
- [StableDrag: Stable Dragging for Point-based Image Editing](https://arxiv.org/abs/2403.04437), ECCV, 2024
- [StyleTokenizer: Defining Image Style by a Single Instance for Controlling Diffusion Models](https://arxiv.org/abs/2409.02543), ECCV, 2024 | [code](https://github.com/alipay/style-tokenizer)
- [Taming Latent Diffusion Model for Neural Radiance Field Inpainting](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/354_ECCV_2024_paper.php), ECCV, 2024
- [TinyBeauty: Toward Tiny and High-quality Facial Makeup with Data Amplify Learning](https://arxiv.org/abs/2403.15033), ECCV, 2024 | [code](https://github.com/TinyBeauty/TinyBeauty)
- [Tuning-Free Image Customization with Image and Text Guidance](https://arxiv.org/abs/2403.12658), ECCV, 2024
- [TurboEdit: Instant text-based image editing](https://arxiv.org/abs/2408.08332), ECCV, 2024
- [Watch Your Steps: Local Image and Scene Editing by Text Instructions](https://arxiv.org/abs/2308.08947), ECCV, 2024 | [code](https://github.com/SamsungLabs/WatchYourSteps)
- [3D-Aware Face Editing via Warping-Guided Latent Direction Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_3D-Aware_Face_Editing_via_Warping-Guided_Latent_Direction_Learning_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/cyh-sj/FaceEdit3D)
- [An Edit Friendly DDPM Noise Space: Inversion and Manipulations](https://arxiv.org/abs/2304.06140), CVPR, 2024 | [code](https://github.com/inbarhub/DDPM_inversion)
- [Benchmarking Segmentation Models with Mask-Preserved Attribute Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_Benchmarking_Segmentation_Models_with_Mask-Preserved_Attribute_Editing_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/PRIS-CV/Pascal-EA)
- [Choose What You Need: Disentangled Representation Learning for Scene Text Recognition Removal and Editing](https://arxiv.org/abs/2405.04377), CVPR, 2024
- [Content-Style Decoupling for Unsupervised Makeup Transfer without Generating Pseudo Ground Truth](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Content-Style_Decoupling_for_Unsupervised_Makeup_Transfer_without_Generating_Pseudo_Ground_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/Snowfallingplum/CSD-MT)
- [Contrastive Denoising Score for Text-guided Latent Diffusion Image Editing](https://arxiv.org/abs/2311.18608), CVPR, 2024 | [code](https://github.com/HyelinNAM/ContrastiveDenoisingScore)
- [DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations](https://arxiv.org/abs/2403.06951), CVPR, 2024 | [code](https://github.com/Tianhao-Qi/DEADiff_code)
- [Deformable One-shot Face Stylization via DINO Semantic Guidance](https://arxiv.org/abs/2403.00459), CVPR, 2024 | [code](https://github.com/zichongc/DoesFS)
- [DemoCaricature: Democratising Caricature Generation with a Rough Sketch](https://arxiv.org/abs/2312.04364), CVPR, 2024 | [code](https://github.com/ChenDarYen/DemoCaricature)
- [DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_DiffAM_Diffusion-based_Adversarial_Makeup_Transfer_for_Facial_Privacy_Protection_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/HansSunY/DiffAM)
- [DiffEditor: Boosting Accuracy and Flexibility on Diffusion-based Image Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Mou_DiffEditor_Boosting_Accuracy_and_Flexibility_on_Diffusion-based_Image_Editing_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/MC-E/DragonDiffusion)
- [DiffMorpher: Unleashing the Capability of Diffusion Models for Image Morphing](https://arxiv.org/abs/2312.07409), CVPR, 2024 | [code](https://github.com/Kevin-thu/DiffMorpher)
- [Diffusion Handles: Enabling 3D Edits for Diffusion Models by Lifting Activations to 3D](https://arxiv.org/abs/2312.02190), CVPR, 2024 | [code](https://github.com/adobe-research/DiffusionHandles)
- [DiffusionLight: Light Probes for Free by Painting a Chrome Ball](https://arxiv.org/abs/2312.09168), CVPR, 2024 | [code](https://github.com/DiffusionLight/DiffusionLight)
- [Distraction is All You Need: Memory-Efficient Image Immunization against Diffusion-Based Image Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Lo_Distraction_is_All_You_Need_Memory-Efficient_Image_Immunization_against_Diffusion-Based_CVPR_2024_paper.html), CVPR, 2024
- [Doubly Abductive Counterfactual Inference for Text-based Image Editing](https://arxiv.org/abs/2403.02981), CVPR, 2024 | [code](https://github.com/xuesong39/DAC)
- [Edit One for All: Interactive Batch Image Editing](https://arxiv.org/abs/2401.10219), CVPR, 2024 | [code](https://github.com/thaoshibe/edit-one-for-all)
- [Emu Edit: Precise Image Editing via Recognition and Generation Tasks](https://openaccess.thecvf.com/content/CVPR2024/html/Sheynin_Emu_Edit_Precise_Image_Editing_via_Recognition_and_Generation_Tasks_CVPR_2024_paper.html), CVPR, 2024
- [Face2Diffusion for Fast and Editable Face Personalization](https://arxiv.org/abs/2403.05094), CVPR, 2024 | [code](https://github.com/mapooon/Face2Diffusion)
- [Focus on Your Instruction: Fine-grained and Multi-instruction Image Editing by Attention Modulation](https://arxiv.org/abs/2312.10113), CVPR, 2024 | [code](https://github.com/guoqincode/Focus-on-Your-Instruction)
- [FreeDrag: Feature Dragging for Reliable Point-based Image Editing](https://arxiv.org/abs/2307.04684), CVPR, 2024 | [code](https://github.com/LPengYang/FreeDrag)
- [HIVE: Harnessing Human Feedback for Instructional Visual Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_HIVE_Harnessing_Human_Feedback_for_Instructional_Visual_Editing_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/salesforce/HIVE)
- [Holo-Relighting: Controllable Volumetric Portrait Relighting from a Single Image](https://arxiv.org/abs/2403.09632), CVPR, 2024 | [code](https://github.com/guoqincode/Focus-on-Your-Instruction)
- [IDGuard: Robust General Identity-centric POI Proactive Defense Against Face Editing Abuse](https://openaccess.thecvf.com/content/CVPR2024/html/Dai_IDGuard_Robust_General_Identity-centric_POI_Proactive_Defense_Against_Face_Editing_CVPR_2024_paper.html), CVPR, 2024
- [Image Sculpting: Precise Object Editing with 3D Geometry Control](https://arxiv.org/abs/2401.01702), CVPR, 2024 | [code](https://github.com/vision-x-nyu/image-sculpting)
- [In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editing](https://arxiv.org/abs/2312.04965), CVPR, 2024 | [code](https://github.com/Twizwei/in-n-out)
- [Inversion-Free Image Editing with Language-Guided Diffusion Models](https://arxiv.org/abs/2312.04965), CVPR, 2024 | [code](https://github.com/sled-group/InfEdit)
- [LEDITS++: Limitless Image Editing using Text-to-Image Models](https://openaccess.thecvf.com/content/CVPR2024/html/Brack_LEDITS_Limitless_Image_Editing_using_Text-to-Image_Models_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/ml-research/ledits_pp)
- [M&M VTO: Multi-Garment Virtual Try-On and Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_MM_VTO_Multi-Garment_Virtual_Try-On_and_Editing_CVPR_2024_paper.html), CVPR, 2024
- [PAIR-Diffusion: Object-Level Image Editing with Structure-and-Appearance Paired Diffusion Models](https://arxiv.org/abs/2303.17546), CVPR, 2024 | [code](https://github.com/Picsart-AI-Research/PAIR-Diffusion)
- [Person in Place: Generating Associative Skeleton-Guidance Maps for Human-Object Interaction Image Editing](https://arxiv.org/abs/2303.17546), CVPR, 2024 | [code](https://github.com/YangChangHee/CVPR2024_Person-In-Place_RELEASE)
- [Puff-Net: Efficient Style Transfer with Pure Content and Style Feature Fusion Network](https://arxiv.org/abs/2405.19775), CVPR, 2024
- [PIA: Your Personalized Image Animator via Plug-and-Play Modules in Text-to-Image Models](https://arxiv.org/abs/2312.13964), CVPR, 2024 | [code](https://github.com/open-mmlab/PIA)
- [Referring Image Editing: Object-level Image Editing via Referring Expressions](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Referring_Image_Editing_Object-level_Image_Editing_via_Referring_Expressions_CVPR_2024_paper.html), CVPR, 2024
- [RealCustom: Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization](https://arxiv.org/abs/2403.00483), CVPR, 2024
- [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_SCEdit_Efficient_and_Controllable_Image_Diffusion_Generation_via_Skip_Connection_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/ali-vilab/SCEdit)
- [SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal Large Language Models](https://arxiv.org/abs/2312.06739), CVPR, 2024 | [code](https://github.com/TencentARC/SmartEdit)
- [Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer](https://arxiv.org/abs/2312.09008), CVPR, 2024 | [code](https://github.com/jiwoogit/StyleID)
- [SwitchLight: Co-design of Physics-driven Architecture and Pre-training Framework for Human Portrait Relighting](https://arxiv.org/abs/2402.18848), CVPR, 2024
- [Text-Driven Image Editing via Learnable Regions](https://arxiv.org/abs/2311.16432), CVPR, 2024 | [code](https://github.com/yuanze-lin/Learnable_Regions)
- [Texture-Preserving Diffusion Models for High-Fidelity Virtual Try-On](https://arxiv.org/abs/2404.01089), CVPR, 2024 | [code](https://github.com/Gal4way/TPD)
- [The Devil is in the Details: StyleFeatureEditor for Detail-Rich StyleGAN Inversion and High Quality Image Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Bobkov_The_Devil_is_in_the_Details_StyleFeatureEditor_for_Detail-Rich_StyleGAN_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/FusionBrainLab/StyleFeatureEditor)
- [TiNO-Edit: Timestep and Noise Optimization for Robust Diffusion-Based Image Editing](https://arxiv.org/abs/2404.11120), CVPR, 2024 | [code](https://github.com/SherryXTChen/TiNO-Edit)
- [ToonerGAN: Reinforcing GANs for Obfuscating Automated Facial Indexing](https://openaccess.thecvf.com/content/CVPR2024/html/Thakral_ToonerGAN_Reinforcing_GANs_for_Obfuscating_Automated_Facial_Indexing_CVPR_2024_paper.html), CVPR, 2024
- [Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Towards_Understanding_Cross_and_Self-Attention_in_Stable_Diffusion_for_Text-Guided_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/alibaba/EasyNLP/tree/master/diffusion/FreePromptEditing)
- [UniHuman: A Unified Model For Editing Human Images in the Wild](https://arxiv.org/abs/2312.14985), CVPR, 2024 | [code](https://github.com/NannanLi999/UniHuman)
- [Z*: Zero-shot Style Transfer via Attention Reweighting](https://openaccess.thecvf.com/content/CVPR2024/html/Deng_Z_Zero-shot_Style_Transfer_via_Attention_Reweighting_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/HolmesShuan/Zero-shot-Style-Transfer-via-Attention-Rearrangement)
- [ZONE: Zero-Shot Instruction-Guided Local Editing](https://arxiv.org/abs/2312.16794), CVPR, 2024 | [code](https://github.com/lsl001006/ZONE)

</details>

### Video Editing

<details><summary>2024</summary>

- [Be-Your-Outpainter: Mastering Video Outpainting through Input-Specific Adaptation](https://arxiv.org/abs/2403.13745), ECCV, 2024 | [code](https://github.com/G-U-N/Be-Your-Outpainter)
- [DNI: Dilutional Noise Initialization for Diffusion Video Editing](https://arxiv.org/abs/2409.13037), ECCV, 2024
- [DragAnything: Motion Control for Anything using Entity Representation](https://arxiv.org/abs/2403.07420), ECCV, 2024 | [code](https://github.com/showlab/DragAnything)
- [DreamMotion: Space-Time Self-Similar Score Distillation for Zero-Shot Video Editing](https://arxiv.org/abs/2403.12002), ECCV, 2024
- [DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](https://arxiv.org/abs/2409.09605), ECCV, 2024 | [code](https://github.com/leoShen917/DreamMover)
- [Fast Sprite Decomposition from Animated Graphics](https://arxiv.org/abs/2408.03923), ECCV, 2024 | [code](https://github.com/CyberAgentAILab/sprite-decompose)
- [MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2738_ECCV_2024_paper.php), ECCV, 2024
- [TCAN: Animating Human Images with Temporally Consistent Pose Guidance using Diffusion Models](https://arxiv.org/abs/2407.09012), ECCV, 2024 | [code](https://github.com/eccv2024tcan/TCAN)
- [Towards Model-Agnostic Dataset Condensation by Heterogeneous Models](https://arxiv.org/abs/2409.14340), ECCV, 2024 | [code](https://github.com/Tinglok/avsoundscape)
- [WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2554_ECCV_2024_paper.php), ECCV, 2024
- [A Video is Worth 256 Bases: Spatial-Temporal Expectation-Maximization Inversion for Zero-Shot Video Editing](https://arxiv.org/abs/2312.05856), CVPR, 2024 | [code](https://github.com/STEM-Inv/stem-inv)
- [CAMEL: Causal Motion Enhancement tailored for lifting text-driven video editing](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_CAMEL_CAusal_Motion_Enhancement_Tailored_for_Lifting_Text-driven_Video_Editing_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/zhangguiwei610/CAMEL)
- [CCEdit: Creative and Controllable Video Editing via Diffusion Models](https://arxiv.org/abs/2309.16496), CVPR, 2024 | [code](https://github.com/RuoyuFeng/CCEdit)
- [CoDeF: Content Deformation Fields for Temporally Consistent Video Processing](https://arxiv.org/abs/2308.07926), CVPR, 2024 | [code](https://github.com/qiuyu96/CoDeF)
- [DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_DynVideo-E_Harnessing_Dynamic_NeRF_for_Large-Scale_Motion-_and_View-Change_Human-Centric_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/qiuyu96/CoDeF)
- [FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation](https://arxiv.org/abs/2403.12962), CVPR, 2024 | [code](https://github.com/williamyang1991/FRESCO/tree/main)
- [MaskINT: Video Editing via Interpolative Non-autoregressive Masked Transformers](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_MaskINT_Video_Editing_via_Interpolative_Non-autoregressive_Masked_Transformers_CVPR_2024_paper.html), CVPR, 2024
- [RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models](https://arxiv.org/abs/2312.04524), CVPR, 2024 | [code](https://github.com/rehg-lab/RAVE)
- [VidToMe: Video Token Merging for Zero-Shot Video Editing](https://arxiv.org/abs/2312.10656), CVPR, 2024 | [code](https://github.com/lixirui142/VidToMe)
- [VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models](https://arxiv.org/abs/2312.00845), CVPR, 2024 | [code](https://github.com/HyeonHo99/Video-Motion-Customization)

</details>

### 3D Editing  

<details><summary>2024</summary>

- [3DEgo: 3D Editing on the Go!](https://arxiv.org/abs/2407.10102), ECCV, 2024
- [Chat-Edit-3D: Interactive 3D Scene Editing via Text Prompts](https://arxiv.org/abs/2407.06842), ECCV, 2024 | [code](https://github.com/Fangkang515/CE3D)
- [DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing](https://arxiv.org/abs/2404.18929), ECCV, 2024 | [code](https://github.com/silent-chen/DGE)
- [Free-Editor: Zero-shot Text-driven 3D Scene Editing](https://arxiv.org/abs/2312.13663), ECCV, 2024 | [code](https://github.com/nazmul-karim170/FreeEditor-Text-to-3D-Scene-Editing)
- [GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](https://arxiv.org/abs/2403.08733), ECCV, 2024 | [code](https://github.com/ActiveVisionLab/gaussctrl)
- [Gaussian Grouping: Segment and Edit Anything in 3D Scenes](https://arxiv.org/abs/2312.00732), ECCV, 2024 | [code](https://github.com/lkeab/gaussian-grouping)
- [KMTalk: Speech-Driven 3D Facial Animation with Key Motion Embedding](https://arxiv.org/abs/2409.01113), ECCV, 2024 | [code](https://github.com/ffxzh/KMTalk)
- [LatentEditor: Text Driven Local Editing of 3D Scenes](https://arxiv.org/abs/2312.09313), ECCV, 2024 | [code](https://github.com/umarkhalidAI/LatentEditor)
- [RoomTex: Texturing Compositional Indoor Scenes via Iterative Inpainting](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8662_ECCV_2024_paper.php), ECCV, 2024 | [code](https://github.com/qwang666/RoomTex-)
- [SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer](https://arxiv.org/abs/2403.18512), ECCV, 2024 | [code](https://github.com/JarrentWu1031/SC4D)
- [Shapefusion: 3D localized human diffusion models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2155_ECCV_2024_paper.php), ECCV, 2024
- [SMooDi: Stylized Motion Diffusion Model](https://arxiv.org/abs/2407.12783), ECCV, 2024 | [code](https://github.com/neu-vi/SMooDi)
- [StyleCity: Large-Scale 3D Urban Scenes Stylization](https://arxiv.org/abs/2404.10681), ECCV, 2024 | [code](https://github.com/chenyingshu/stylecity3d)
- [Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](https://arxiv.org/abs/2403.10050), ECCV, 2024 | [code](https://github.com/slothfulxtx/Texture-GS)
- [Towards High-Quality 3D Motion Transfer with Realistic Apparel Animation](https://arxiv.org/abs/2407.11266), ECCV, 2024 | [code](https://github.com/rongakowang/MMDMC)
- [View-Consistent 3D Editing with Gaussian Splatting](https://arxiv.org/abs/2403.11868), ECCV, 2024 | [code](https://github.com/Yuxuan-W/vcedit)
- [Watch Your Steps: Local Image and Scene Editing by Text Instructions](https://arxiv.org/abs/2308.08947), ECCV, 2024 | [code](https://github.com/SamsungLabs/WatchYourSteps)
- [Arbitrary Motion Style Transfer with Multi-condition Motion Latent Diffusion Model](https://openaccess.thecvf.com/content/CVPR2024/html/Song_Arbitrary_Motion_Style_Transfer_with_Multi-condition_Motion_Latent_Diffusion_Model_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/XingliangJin/MCM-LDM)
- [ConsistDreamer: 3D-Consistent 2D Diffusion for High-Fidelity Scene Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_ConsistDreamer_3D-Consistent_2D_Diffusion_for_High-Fidelity_Scene_Editing_CVPR_2024_paper.html), CVPR, 2024
- [Control4D: Efficient 4D Portrait Editing with Text](https://openaccess.thecvf.com/content/CVPR2024/html/Shao_Control4D_Efficient_4D_Portrait_Editing_with_Text_CVPR_2024_paper.html), CVPR, 2024
- [Customize your NeRF: Adaptive Source Driven 3D Scene Editing via Local-Global Iterative Training](https://openaccess.thecvf.com/content/CVPR2024/html/He_Customize_your_NeRF_Adaptive_Source_Driven_3D_Scene_Editing_via_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/hrz2000/CustomNeRF)
- [GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting](https://arxiv.org/abs/2311.14521), CVPR, 2024 | [code](https://github.com/buaacyw/GaussianEditor)
- [GeneAvatar: Generic Expression-Aware Volumetric Head Avatar Editing from a Single Image](https://openaccess.thecvf.com/content/CVPR2024/html/Bao_GeneAvatar_Generic_Expression-Aware_Volumetric_Head_Avatar_Editing_from_a_Single_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/zju3dv/GeneAvatar)
- [GenN2N: Generative NeRF2NeRF Translation](https://arxiv.org/abs/2404.02788), CVPR, 2024 | [code](https://github.com/Lxiangyue/GenN2N)
- [Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion](https://openaccess.thecvf.com/content/CVPR2024/html/Mou_Instruct_4D-to-4D_Editing_4D_Scenes_as_Pseudo-3D_Scenes_Using_2D_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/Friedrich-M/Instruct-4D-to-4D)
- [LAENeRF: Local Appearance Editing for Neural Radiance Fields](https://openaccess.thecvf.com/content/CVPR2024/html/Radl_LAENeRF_Local_Appearance_Editing_for_Neural_Radiance_Fields_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/r4dl/LAENeRF)
- [Makeup Prior Models for 3D Facial Makeup Estimation and Applications](https://arxiv.org/abs/2403.17761), CVPR, 2024 | [code](https://github.com/YangXingchao/makeup-priors)
- [SHAP-EDITOR: Instruction-Guided Latent 3D Editing in Seconds](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_SHAP-EDITOR_Instruction-Guided_Latent_3D_Editing_in_Seconds_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/silent-chen/Shap-Editor)
- [ShapeWalk: Compositional Shape Editing through Language-Guided Chains](https://openaccess.thecvf.com/content/CVPR2024/html/Slim_ShapeWalk_Compositional_Shape_Editing_Through_Language-Guided_Chains_CVPR_2024_paper.html), CVPR, 2024
- [StrokeFaceNeRF: Stroke-based Facial Appearance Editing in Neural Radiance Field](https://openaccess.thecvf.com/content/CVPR2024/html/Li_StrokeFaceNeRF_Stroke-based_Facial_Appearance_Editing_in_Neural_Radiance_Field_CVPR_2024_paper.html), CVPR, 2024
- [Text-Guided 3D Face Synthesis - From Generation to Editing](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Text-Guided_3D_Face_Synthesis_-_From_Generation_to_Editing_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/JiejiangWu/FaceG2E)
</details>

## Multimodal

<details><summary>2024</summary>

- [Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model](https://arxiv.org/abs/2402.19150), ECCV, 2024 | [code](https://github.com/ChaduCheng/TypoDeceptions)
- [AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](https://arxiv.org/abs/2403.09513), ECCV, 2024 | [code](https://github.com/SaFoLab-WISC/AdaShield)
- [AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](https://arxiv.org/abs/2407.08156), ECCV, 2024 | [code](https://github.com/xsx1001/AddressCLIP)
- [Adversarial Prompt Tuning for Vision-Language Models](https://arxiv.org/abs/2403.12002), ECCV, 2024 | [code](https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning)
- [A Large Multimodal Model Perceiving Any Aspect Ratio and High-Resolution Images](https://arxiv.org/abs/2403.11703), ECCV, 2024 | [code](https://github.com/thunlp/LLaVA-UHD)
- [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models](https://arxiv.org/abs/2403.06764), ECCV, 2024 | [code](https://github.com/pkunlp-icler/FastV)
- [API: Attention Prompting on Image for Large Vision-Language Models](https://arxiv.org/abs/2403.12658), ECCV, 2024 | [code](https://github.com/yu-rp/apiprompting)
- [Bi-directional Contextual Attention for 3D Dense Captioning](https://arxiv.org/abs/2408.06662), ECCV, 2024
- [BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation](https://arxiv.org/abs/2408.05926), ECCV, 2024
- [CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts](https://arxiv.org/abs/2311.16445), ECCV, 2024
- [ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference](https://arxiv.org/abs/2407.12442), ECCV, 2024 | [code](https://github.com/mc-lan/ClearCLIP)
- [ControlCap: Controllable Region-level Captioning](https://arxiv.org/abs/2401.17910), ECCV, 2024 | [code](https://github.com/callsys/ControlCap)
- [Controllable Navigation Instruction Generation with Chain of Thought Prompting](https://arxiv.org/abs/2407.07433), ECCV, 2024
- [DreamLIP: Language-Image Pre-training with Long Captions](https://arxiv.org/abs/2403.17007), ECCV, 2024 | [code](https://github.com/zyf0619sjtu/DreamLIP)
- [DriveLM: Driving with Graph Visual Question Answering](https://arxiv.org/abs/2312.14150), ECCV, 2024 | [code](https://github.com/OpenDriveLab/DriveLM)
- [Elysium: Exploring Object-level Perception in Videos via MLLM](https://arxiv.org/abs/2403.16558), ECCV, 2024 | [code](https://github.com/Hon-Wong/Elysium)
- [Emergent Visual-Semantic Hierarchies in Image-Text Representations](https://arxiv.org/abs/2407.08521), ECCV, 2024 | [code](https://github.com/TAU-VAILab/hierarcaps)
- [Empowering Multimodal Large Language Model as a Powerful Data Generator](https://arxiv.org/abs/2409.01322), ECCV, 2024 | [code](https://github.com/zhaohengyuan1/Genixer)
- [EventBind: Learning a Unified Representation to Bind Them All for Event-based Open-world Understanding](https://arxiv.org/abs/2308.03135), ECCV, 2024 | [code](https://github.com/jiazhou-garland/EventBind)
- [FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance](https://arxiv.org/abs/2407.05578), ECCV, 2024 | [code](https://github.com/pumpkin805/FALIP)
- [Alpha-CLIP: A CLIP Model Focusing on Wherever You Want](https://arxiv.org/abs/2312.03818), CVPR, 2024. | [code](https://github.com/SunzeY/AlphaCLIP)
- [Anchor-based Robust Finetuning of Vision-Language Models](https://arxiv.org/abs/2404.06244), CVPR, 2024. | [code](https://github.com/LixDemon/ARF)
- [BioCLIP: A Vision Foundation Model for the Tree of Life](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/Imageomics/bioclip)
- [BioCLIP: A Vision Foundation Model for the Tree of Life](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/Imageomics/bioclip)
- [Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters](https://arxiv.org/abs/2403.11549), CVPR, 2024 | [code](https://github.com/JiazuoYu/MoE-Adapters4CL)
- [Can Language Beat Numerical Regression? Language-Based Multimodal Trajectory Prediction](https://arxiv.org/abs/2403.18447), CVPR, 2024 | [code](https://github.com/InhwanBae/LMTrajectory)
- [Can't make an Omelette without Breaking some Eggs: Plausible Action Anticipation using Large Video-Language Models](https://arxiv.org/abs/2405.20305), CVPR, 2024
- [Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding](https://arxiv.org/abs/2311.08046), CVPR, 2024 | [code](https://github.com/PKU-YuanGroup/Chat-UniVi)
- [Compositional Chain-of-Thought Prompting for Large Multimodal Models](https://arxiv.org/abs/2311.17076), CVPR, 2024 | [code](https://github.com/chancharikmitra/CCoT)
- [Describing Differences in Image Sets with Natural Language](https://arxiv.org/abs/2312.02974), CVPR, 2024 | [code](https://github.com/Understanding-Visual-Datasets/VisDiff)
- [Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models](https://arxiv.org/abs/2403.17589), CVPR, 2024 | [code](https://github.com/YBZh/DMN)
- [Efficient Stitchable Task Adaptation](https://arxiv.org/abs/2311.17352), CVPR, 2024 | [code](https://github.com/ziplab/Stitched_LLaMA)
- [Efficient Test-Time Adaptation of Vision-Language Models](https://arxiv.org/abs/2403.18293), CVPR, 2024 | [code](https://github.com/kdiAAA/TDA)
- [Exploring the Transferability of Visual Prompting for Multimodal Large Language Models](https://arxiv.org/abs/2404.11207), CVPR, 2024 | [code](https://github.com/zycheiheihei/transferable-visual-prompting)
- [FairCLIP: Harnessing Fairness in Vision-Language Learning](https://arxiv.org/abs/2403.19949), CVPR, 2024 | [code](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP)
- [FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication](https://arxiv.org/abs/2404.16123), CVPR, 2024
- [FFF: Fixing Flawed Foundations in contrastive pre-training results in very strong Vision-Language models](https://arxiv.org/abs/2404.16123), CVPR, 2024
- [Generative Multimodal Models are In-Context Learners](https://arxiv.org/abs/2312.13286), CVPR, 2024 | [code](https://github.com/baaivision/Emu/tree/main/Emu2)
- [GLaMM: Pixel Grounding Large Multimodal Model](https://arxiv.org/abs/2311.03356), CVPR, 2024 | [code](https://github.com/mbzuai-oryx/groundingLMM)
- [GPT4Point: A Unified Framework for Point-Language Understanding and Generation](https://arxiv.org/abs/2312.02980), CVPR, 2024 | [code](https://github.com/Pointcept/GPT4Point)
- [Improved Baselines with Visual Instruction Tuning](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/haotian-liu/LLaVA)
- [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238), CVPR, 2024 | [code](https://github.com/OpenGVLab/InternVL)
- [Learning by Correction: Efficient Tuning Task for Zero-Shot Generative Vision-Language Reasoning](https://arxiv.org/abs/2404.00909), CVPR, 2024
- [Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation](https://arxiv.org/abs/2312.02439), CVPR, 2024 | [code](https://github.com/sail-sg/CLoT)
- [LION : Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge](https://arxiv.org/abs/2311.11860), CVPR, 2024 | [code](https://github.com/rshaojimmy/JiuTian)
- [LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning](https://arxiv.org/abs/2311.18651), CVPR, 2024 | [code](https://github.com/Open3DA/LL3DA)
- [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922), CVPR, 2024 | [code](https://github.com/DAMO-NLP-SG/VCD)
- [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://arxiv.org/abs/2311.16502), CVPR, 2024 | [code](https://github.com/MMMU-Benchmark/MMMU)
- [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://arxiv.org/abs/2311.17049), CVPR, 2024 | [code](https://github.com/apple/ml-mobileclip)
- [MoPE-CLIP: Structured Pruning for Efficient Vision-Language Models with Module-wise Pruning Error Metric](https://arxiv.org/abs/2403.07839), CVPR, 2024
- [Narrative Action Evaluation with Prompt-Guided Multimodal Interaction](https://arxiv.org/abs/2404.14471), CVPR, 2024 | [code](https://github.com/shiyi-zh0408/NAE_CVPR2024)
- [OneLLM: One Framework to Align All Modalities with Language](https://arxiv.org/abs/2312.03700), CVPR, 2024 | [code](https://github.com/csuhan/OneLLM)
- [One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models](https://arxiv.org/abs/2403.01849), CVPR, 2024 | [code](https://github.com/TreeLLi/APT)
- [OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation](https://arxiv.org/abs/2402.19479), CVPR, 2024 | [code](https://github.com/shikiw/OPERA)
- [Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers](https://arxiv.org/abs/2311.17911), CVPR, 2024 | [code](https://github.com/snap-research/Panda-70M)
- [PixelLM: Pixel Reasoning with Large Multimodal Model](https://arxiv.org/abs/2312.02228), CVPR, 2024 | [code](https://github.com/MaverickRen/PixelLM)
- [PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization](https://arxiv.org/abs/2404.09011), CVPR, 2024
- [Prompt Highlighter: Interactive Control for Multi-Modal LLMs](https://arxiv.org/abs/2312.04302), CVPR, 2024 | [code](https://github.com/dvlab-research/Prompt-Highlighter)
- [PromptKD: Unsupervised Prompt Distillation for Vision-Language Models](https://arxiv.org/abs/2403.02781), CVPR, 2024 | [code](https://github.com/zhengli97/PromptKD)
- [Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models](https://arxiv.org/abs/2311.06783), CVPR, 2024 | [code](https://github.com/Q-Future/Q-Instruct)
- [SC-Tune: Unleashing Self-Consistent Referential Comprehension in Large Vision Language Models](https://arxiv.org/abs/2403.13263), CVPR, 2024 | [code](https://github.com/ivattyue/SC-Tune)
- [SEED-Bench: Benchmarking Multimodal Large Language Models](https://arxiv.org/abs/2311.17092), CVPR, 2024 | [code](https://github.com/AILab-CVC/SEED-Bench)
- [SyncMask: Synchronized Attentional Masking for Fashion-centric Vision-Language Pretraining](https://arxiv.org/abs/2404.01156), CVPR, 2024
- [The Manga Whisperer: Automatically Generating Transcriptions for Comics](https://arxiv.org/abs/2401.10224), CVPR, 2024 | [code](https://github.com/ragavsachdeva/magi)
- [UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All](https://arxiv.org/abs/2403.12532), CVPR, 2024
- [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982), CVPR, 2024 | [code](https://github.com/Vchitect/VBench)
- [VideoChat: Chat-Centric Video Understanding](https://arxiv.org/abs/2305.06355), CVPR, 2024 | [code](https://github.com/OpenGVLab/Ask-Anything)
- [ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/abs/2312.00784), CVPR, 2024 | [code](https://github.com/mu-cai/ViP-LLaVA)
- [ViTamin: Designing Scalable Vision Models in the Vision-language Era](https://arxiv.org/abs/2404.02132), CVPR, 2024 | [code](https://github.com/Beckschen/ViTamin)
- [ViT-Lens: Towards Omni-modal Representations](https://github.com/TencentARC/ViT-Lens), CVPR, 2024 | [code](https://arxiv.org/abs/2308.10185)

</details>

## Others

<details><summary>2024</summary>

- [Which Model Generated This Image? A Model-Agnostic Approach for Origin Attribution](https://arxiv.org/abs/2404.02697v2), ECCV, 2024 | [code](https://github.com/uwFengyuan/OCC-CLIP)
- [AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error](https://arxiv.org/abs/2401.17879), CVPR, 2024 | [code](https://github.com/jonasricker/aeroblade)
- [Diff-BGM: A Diffusion Model for Video Background Music Generation](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Diff-BGM_A_Diffusion_Model_for_Video_Background_Music_Generation_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/sizhelee/Diff-BGM)
- [EvalCrafter: Benchmarking and Evaluating Large Video Generation Models](https://arxiv.org/abs/2310.11440), CVPR, 2024 | [code](https://github.com/evalcrafter/EvalCrafter)
- [FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_FlashEval_Towards_Fast_and_Accurate_Evaluation_of_Text-to-image_Diffusion_Generative_CVPR_2024_paper.html), CVPR, 2024
- [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/liangyanshuo/InfLoRA)
- [On the Content Bias in Fréchet Video Distance](https://arxiv.org/abs/2404.12391), CVPR, 2024 | [code](https://github.com/songweige/content-debiased-fvd)
- [Shadows Don’t Lie and Lines Can’t Bend! Generative Models don’t know Projective Geometry...for now](https://openaccess.thecvf.com/content/CVPR2024/html/Sarkar_Shadows_Dont_Lie_and_Lines_Cant_Bend_Generative_Models_dont_CVPR_2024_paper.html), CVPR, 2024 | [code](https://github.com/hanlinm2/projective-geometry)
- [TexTile: A Differentiable Metric for Texture Tileability](https://arxiv.org/abs/2403.12961v1), CVPR, 2024 | [code](https://github.com/crp94/textile)

</details>

## Codebase

### Image Synthesis

- [Flux.1](https://huggingface.co/black-forest-labs) | [code](https://github.com/black-forest-labs/flux) ![Github Repo stars]((https://img.shields.io/github/stars/black-forest-labs/flux)

### Video Synthesis

- [HunyuanVideo: A Systematic Framework For Large Video Generation Model Training](https://github.com/Tencent/HunyuanVideo/blob/main/assets/hunyuanvideo.pdf) | [code](https://github.com/Tencent/HunyuanVideo) ![GitHub Repo stars](https://img.shields.io/github/stars/Tencent/HunyuanVideo)
- LTX-Video: A Lightweight Video Transformer for Efficient Video Generation [[code](https://github.com/Lightricks/LTX-Video)], [model](https://huggingface.co/Lightricks/LTX-Video) ![GitHub Repo stars](https://img.shields.io/github/stars/Lightricks/LTX-Video)
- [Mochi-1](https://www.genmo.ai/blog) | [code](https://github.com/genmoai/mochi) ![GitHub Repo stars](https://img.shields.io/github/stars/genmoai/mochi)
- [CogVideo](https://arxiv.org/abs/2408.06072) | [code](https://github.com/THUDM/CogVideo) ![GitHub Repo stars](https://img.shields.io/github/stars/THUDM/CogVideo)

## Leadboard

- [VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)

## Links

- [NeurIPS2024 Accepted Papers](https://neurips.cc/virtual/2024/papers.html?filter=titles)
- [ECCV2024 Accepted Papers](https://docs.google.com/spreadsheets/d/1G8FQNlitoRr1oK2-LZEloeg0_VBP-E0J_WoSXqAhxNo/pubhtml#)
- [CVPR2024 Accepted Papers](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers)
- [SIGGRAPH Asia2024 Accepted Papers](https://www.realtimerendering.com/kesen/siga2024Papers.htm)
- [SIGGRAPH2024 Accepted Papers](https://www.realtimerendering.com/kesen/sig2024.html)
- [Papercopilot](https://papercopilot.com/statistics)