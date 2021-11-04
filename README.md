# FLOPs-Computation-For-DETRseries
## Introduction
This repo is inherited from [official FLOPs computation for DETR](https://gist.github.com/fmassa/c0fbb9fe7bf53b533b5cc241f5c8234c). Compared with official scripts, this repo adds the FLOPs and the number of params computation of some submodules, which are encoder-decoder, encoder and decoder. 
The script files, including flop_count.py and jit_handles.py, are shared across all models.
## Env
These scripts is tested on PyToch1.8.
## Usage
For DETR, put flop_count.py, jit_handles.py and detr_compute_flops.py in the root of DETR codebase, and then run ```python detr_compute_flops.py```
******
For Deformable DETR, put flop_count.py, jit_handles.py and deformable_detr_compute_flops.py in the root of Deformable DETR codebase, and then run ```python deformable_detr_compute_flops.py```
******
For ConditionalDETR, put flop_count.py, jit_handles.py and conditional_detr_compute_flops.py in the root of ConditionalDETR codebase, and then run ```python conditional_detr_compute_flops.py```
## For your model based DETR
Please refer to custom_compute_flops.py, and modify it following the comments in the file. And then put flop_count.py, jit_handles.py and custom_compute_flops.py
in the root of your project.
