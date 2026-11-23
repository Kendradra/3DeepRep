# 3DeepRep

Official implementation of **3DeepRep: 3D Deep Low-rank Tensor Representation for Hyperspectral Image Inpainting**.

3DeepRep recovers a hyperspectral image (HSI) from partial observations by jointly learning three directional low-rank latent tensors, mapping each of them back to the image domain with a coupled transform, and fusing the three candidates with a learnable aggregation module. Training is fully self-supervised and gradient-driven: the loss is computed only on observed entries.

> Yunshan Li, Wenwu Gong, Qianqian Wang, Chao Wang, Lili Yang.  
> **3DeepRep: 3D deep low-rank tensor representation for hyperspectral image inpainting.**  
> *Neurocomputing*, 678 (2026) 133160.  
> [https://doi.org/10.1016/j.neucom.2026.133160](https://doi.org/10.1016/j.neucom.2026.133160)

## Method

The figure below is Figure 2 of the paper.

<p align="center">
  <img src="flowchart.jpg" alt="Flowchart of the proposed 3DeepRep framework for HSI inpainting." width="100%">
</p>

**Figure 2.** Flowchart of the proposed 3DeepRep framework for HSI inpainting. The process initiates with three explicit latent variables $\{\hat{\mathcal{X}}_i\}_{i=1}^{3}$, corresponding to the three tensor directions. Low-rank structures are enforced on these variables by minimizing the nuclear norms of their respective frontal slices. Subsequently, these latent representations are mapped to the pixel space via multi-directional deep transforms (implemented by Coupled Transform Blocks), yielding three image candidates $\{\mathcal{X}_i\}_{i=1}^{3}$. Finally, these directional intermediates are adaptively fused by a learnable aggregation module to produce the recovered image $\mathcal{X}$.

Given an observed HSI `O` and a binary mask `Ω`:

1. Maintain three latent tensors, one per mode, initialized from a TNN warm start (`Pavia_init.npy` in this demo).
2. Decode each latent tensor with a directional Coupled Transform Block to obtain image candidates `X1`, `X2`, `X3`.
3. Concatenate the candidates along the spectral mode and decode them with the aggregation block to obtain the restored image `X`.
4. Minimize, with Adam,

   `β * weighted nuclear norms + γ * sum of directional fidelity terms + fidelity of X`,

    where fidelity is the masked MSE between a reconstruction and the observed image.

In this implementation the nuclear norm is applied to **one frontal slice per iteration** (cycled by the iteration index), which approximates the sum over all frontal slices in the paper.

## Repository structure

```text
.
├── 3DeepRep_demo.ipynb    # model definition, training loop, and the demo run
├── utils.py               # band-wise PSNR
├── flowchart.jpg          # framework figure (Figure 2 of the paper)
├── README.md
└── dataset/
    ├── Pavia_gtruth.npy           # Pavia Centre, 200 × 200 × 80, values in (0, 1]
    └── MR_90/
        ├── Pavia_sampled.npy      # observed image (unobserved entries are 0)
        ├── Pavia_mask.npy         # 1 = observed, 0 = missing; sampling rate 0.10
        └── Pavia_init.npy         # TNN warm start used to initialize the latent tensors
```

The demo uses Pavia Centre (`200 × 200 × 80`) with a random missing rate of 0.90. `Pavia_mask.npy` marks observed pixels with 1 (sampling rate 0.10). `Pavia_sampled.npy` keeps those pixels and sets the rest to 0. `Pavia_gtruth.npy` is used only to report PSNR. `Pavia_init.npy` is the TNN warm start copied into the three latent tensors. Pixel values lie in `(0, 1]`. Pavia Centre is distributed by the [Universidad del País Vasco / EHU hyperspectral scenes collection](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

## Requirements

- NVIDIA GPU with CUDA. `utils.py` sets `dtype = torch.cuda.FloatTensor`, so a CPU-only install will fail at import.
- Python 3.10 (the notebook kernel used for development).
- PyTorch, NumPy, SciPy, scikit-image, Matplotlib, pandas, Jupyter.

```bash
pip install torch numpy scipy scikit-image matplotlib pandas jupyter
```

Install a CUDA build of PyTorch that matches your driver: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

## Quick start

From the repository root, open `3DeepRep_demo.ipynb` and run all cells. Data paths, model settings, and training hyperparameters are set in that notebook.

## Citation

```bibtex
@article{li2026_3deeprep,
  title   = {3DeepRep: 3D deep low-rank tensor representation for hyperspectral image inpainting},
  author  = {Li, Yunshan and Gong, Wenwu and Wang, Qianqian and Wang, Chao and Yang, Lili},
  journal = {Neurocomputing},
  volume  = {678},
  pages   = {133160},
  year    = {2026},
  doi     = {10.1016/j.neucom.2026.133160}
}
```
