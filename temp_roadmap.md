
2026Q2 roadmap for multi-model generation RL

### 🛠 Architecture

#### Rollout Engine
- vLLM-Omni enhancement
  - [ ] Cross-request batching verification (omni supported)
  - [ ] TP support (weight sync issue)
  - [ ] DP support
  - [ ] Verify abort/resume api
  - [ ] Support diffusers backend
  - [ ] EPDG disaggregation (tbc)

#### Trainer
- [ ] diffusers FSDP trainer
  - [ ] SP support
- [ ] VeOmni diffusion trainer as default
  - [ ] Verify parallelism: FSDP + SP (QwenImage), EP (HunyuanImage-3.0)
  - [ ] Support diffusers backend
- [ ] Fully async trainer for diffusion models, verification

### 🎨 Model & Algorithm Supports
- Image Gen (X2I)
  - [ ] QwenImage + FlowGRPO-fast (Dense DiT) [P0]
  - [ ] HunyuanImage 3.0 + MixGRPO (MoE DiT)
  - [ ] GLM-Image + FlowGRPO (AR + Diffusion)
- Video Gen (X2V)
  - [ ] Wan2.1/Wan2.2 + FlowGRPO   [P0]
  - [ ] DanceGRPO based on vLLM-Omni rollouter + VeOmni trainer
- Audio Gen (X2A)
  - [ ] PrismAudio (V2A task, Fast-GRPO)  (tbc)
- Unified Und. & Gen. (UMM)
  - [ ] BAGEL + FlowGRPO   [P0]

### 📖 Documentation
- [ ] Quickstart doc for QwenImage FlowGRPO training
- [ ] How to add a model with vLLM-Omni backend
- [ ] FlowGRPO/MixGRPO training algorithm doc (Key Components, Configure, Reference Performance, etc)

### 💠 Ascend NPU
- [ ] NPU verification

