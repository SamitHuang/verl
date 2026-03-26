
Q2 Roadmap for multi-model generation RL support

### Architecture

#### Rollout Engine
- vLLM-Omni enhancement
  - [ ] Cross-request batching support
  - [ ] Data parallel support
  - [ ] EPDG disaggregation (tbc)
- [ ] Add diffusers backend

#### Trainer
- [x] diffusers FSDP trainer
- [ ] VeOmni diffusion trainer as default (
- [ ] Fully async trainer for diffusion

### Model & Algorithm Supports
- Image Gen (X2I)
  - [ ] QwenImage + FlowGRPO-fast (Dense DiT)
  - [ ] HunyuanImage 3.0 + MixGRPO (MoE DiT)
  - [ ] GLM-Image + FlowGRPO (AR + Diffusion)
- Video Gen (X2V)
  - [ ] Wan2.1/Wan2.2 + FlowGRPO
  - [ ] DanceGRPO based on vLLM-Omni rollouter + VeOmni trainer
- Audio Gen (X2A)
  - [ ] PrismAudio (V2A task, Fast-GRPO)  (tbc)
- Unified Und. & Gen. (UMM)
  - [ ] BAGEL + FlowGRPO

### Ascend NPU
- [ ] NPU verification

### Documentation
- [ ] Quickstart doc for FlowGRPO training
- [ ] Algorithms doc for FlowGRPO (Key Components, Configure, Reference Performance, etc)


