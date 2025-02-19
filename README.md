
# Reference-based Controllable Scene Stylization
This repository is for ReGS introduced in the following paper "ReGS:Reference-based Controllable Scene Stylization", NeurIPS2024, [[Link]](https://proceedings.neurips.cc/paper_files/paper/2024/file/076c1fa639a7190e216e734f0a1b3e7b-Paper-Conference.pdf) 


The code is built on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Code release in progress. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Referenced-based scene stylization that edits the appearance based on a content-aligned reference image is an emerging research area. Starting with a pretrained neural radiance field (NeRF), existing methods typically learn a novel appearance that matches the given style. Despite their effectiveness, they inherently suffer from time-consuming volume rendering, and thus are impractical for many real-time applications. In this work, we propose ReGS, which adapts 3D Gaussian Splatting (3DGS) for reference-based stylization to enable real-time stylized view synthesis. Editing the appearance of a pretrained 3DGS is challenging as it uses discrete Gaussians as 3D representation, which tightly bind appearance with geometry. Simply optimizing the appearance as prior methods do is often insufficient for modeling continuous textures in the given reference image. To address this challenge, we propose a novel texture-guided control mechanism that adaptively adjusts local responsible Gaussians to a new geometric arrangement, serving for desired texture details. The proposed process is guided by texture clues for effective appearance editing, and regularized by scene depth for preserving original geometric structure. With these novel designs, we show ReGs can produce state-of-the-art stylization results that respect the reference texture while embracing real-time rendering speed for free-view navigation.
![ReGS](/Figs/REGS.png)
REGS
## Run the code
1. This codebase takes the pre-trained 3DGS as input and perform stylization through optimization. For the target scene, please run the  [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to reconstruct the scene first. 

2. Prepare the style reference image and original views. We use examples from [ref-npr](https://drive.google.com/drive/folders/1b6L250lrBrSxfKYPmDBHuY_EP9n7WKnA). Put the pre-trained checkpoint, reference, and original scene in the folder following the provided example here (https://drive.google.com/file/d/13IvnbM3aIm5Lrr4IddbrwqSdXjpxSRMa/view?usp=sharing)

3. Stylization
Run the following script to train models.

    **Example command is in the file 'demo.sh'.**

    ```bash
    REF_STY="../refs/flower/flower_blue.png"
    python train_ref.py --eval -s ../refs/flower_llff -m ../output/flower_blue --convert_SHs_python --sh_degree 3 --start_checkpoint ../refs/flower_final.pth --iterations 3000 --densify_until_iter 1500 --ref_img ${REF_STY} --scene_id -1 --densify_grad_threshold 3e-5
    ```
4. Rendering
Run the following script to render stylized novel views.
    
    **Example command is in the file 'demo.sh'.**

      ```bash
     python render.py -m ../output/flower_blue
      ```

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{mei2025regs,
  title={ReGS: Reference-based Controllable Scene Stylization with Gaussian Splatting},
  author={Mei, Yiqun and Xu, Jiacong and Patel, Vishal},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={4035--4061},
  year={2025}
}
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
## Acknowledgements
This code is built on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors for sharing their code.
