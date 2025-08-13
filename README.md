# [2025INFFUS]STR
STR: Spatio-temporal trajectory representation learning with dual-focus encoder for whole trajectory similarity computation

## Conda Dependencies and Required Packages
- python=3.7.12=hf930737_100_cpython
- pytorch=1.11.0=py3.7_cuda11.3_cudnn8.2.0_0
- torchaudio=0.11.0=py37_cu113
- torchvision=0.12.0=py37_cu113
- pandas==1.2.4
- numpy==1.21.5
- trajectory-distance==1.0
- tensorboard==2.5.0

## Running
```bash
python main.py --config=config.yaml --gpu=0
```
The relevant parameters can be modified in the config.yaml file. 

## Citation
If you find anything in this repository useful to your research, please cite our paper :) We sincerely appreciate it.
```
@article{li2025str,
  title={STR: Spatio-temporal trajectory representation learning with dual-focus encoder for whole trajectory similarity computation},
  author={Li, Mengqiu and Niu, Xinzheng and Zhu, Jiahui and Fournier-Viger, Philippe and Wu, Youxi},
  journal={Information Fusion},
  pages={103231},
  year={2025},
  publisher={Elsevier}
}
```
