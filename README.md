# Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy

This repository contains code for the publication _Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy_. See https://hex-lab.io/Hybrid-Foveated-Path-Tracing for more!

There are three main components to this project:
- A path tracer for initial renders and constant foveated image generation
- A Gaussian Splatting training component to optimize peripheral models
- A Unity viewer project

__IMPORTANT:__ *This repository is for reference*. It is not in a state that can be easily run. The components in the subfolders ship with adapted code, and mostly their original readme, which can be somewhat misleading and misrepresent what the code does. You will need to check out what each part does and adapt to your needs. 

## Paper abstract

Volumetric medical imaging offers great potential for understanding complex pathologies. Yet, traditional 2D slices provide little support for interpreting spatial relationships, forcing users to mentally reconstruct anatomy into three dimensions. Direct volumetric path tracing and VR rendering can improve perception but are computationally expensive, while precomputed representations, like Gaussian Splatting, require planning ahead. Both approaches limit interactive use.

We propose a hybrid rendering approach for high-quality, interactive, and immersive anatomical visualization. Our method combines streamed foveated path tracing with a lightweight Gaussian Splatting approximation of the periphery. The peripheral model generation is optimized with volume data and continuously refined using foveal renderings, enabling interactive updates. Depth-guided reprojection further improves robustness to latency and allows users to balance fidelity with refresh rate.

We compare our method against direct path tracing and Gaussian Splatting. Our results highlight how their combination can preserve strengths in visual quality while re-generating the peripheral model in under a second, eliminating extensive preprocessing and approximations. This opens new options for interactive medical visualization.


## Components

### Path tracer

The path tracer is available in the `pathtracing` subfolder. The path tracer is based on https://github.com/nihofm/volren

Code to generate images and run the server is in the scipts subfolder. Note that no volumes are included. See the

### GS training code

The gaussian splatting code is available in the `gaussian` subfolder. It is based on MiniSplatting2, whcih can be found here https://github.com/fatPeter/mini-splatting2

The code uses the depth patch from https://github.com/roth-hex-lab/Multi-Layer-Anatomy-GS-Training useful for training on highly semi-transparent data.

### Unity Viewer

The unity viewer is available in the `unity` subfolder. It is adapted from the Unity Gaussian Splatting project, which can be found here: https://github.com/aras-p/UnityGaussianSplatting/

This project has trained gaussian models and can receive realtime frames to display on top. Note that we can not include the matching volumes, but you can view the trained peripheral models on their own here, or create your own.

Normal Unity conventions apply.
