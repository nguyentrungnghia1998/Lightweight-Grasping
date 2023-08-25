# Antipodal Robotic Grasping
## Installation
- Checkout the robotic grasping package
```bash
$ git clone https://github.com/anavuongdin/robotic-grasping.git
```

- Create a virtual environment
```bash
$ conda create -n grasping python=3.9
```

- Activate the virtual environment
```bash
$ conda activate grasping
```

- Install the requirements
```bash
$ cd robotic-grasping
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Weights
- All weights are stored at `weights/model_<dataset>`.

## Inference example
- An atom example is shown in `run_robotic_exp.py`. To run this file:

```bash
$ python run_robotic_exp.py --weight weights/model_<dataset>
```
