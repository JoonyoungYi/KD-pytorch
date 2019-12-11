# KD-pytorch

* Knowledge Distillation (KD) - pytorch
* PyTorch implementation of [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
* This repository is forked from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
* Dataset: CIFAR10
* Teacher Network: VGG16
* Student Network: CNN with 3 convolutional blocks

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- See `docker/` folder.

## Pretrain Teacher Networks
* Result: 91.90%
* SGD, no weight decay.
* Learning rate adjustment
  * `0.1` for epoch `[0,150)`
  * `0.01` for epoch `[150,250)`
  * `0.001` for epoch `[250,300)`
```
python -m pretrainer --optimizer=sgd --lr=0.1   --start_epoch=0   --n_epoch=150 --model_name=ckpt
python -m pretrainer --optimizer=sgd --lr=0.01  --start_epoch=150 --n_epoch=100 --model_name=ckpt --resume
python -m pretrainer --optimizer=sgd --lr=0.001 --start_epoch=250 --n_epoch=50  --model_name=ckpt --resume
```

## Student Networks
* We use Adam optimizer for fair comparison.
  * max epoch: `300`
  * learning rate: `0.0001`
  * no weight decay for fair comparison.

### Baseline (without Knowledge Distillation)
* Result: 85.01%
```
python -m pretrainer --optimizer=adam --lr=0.0001 --start_epoch=0 --n_epoch=300 --model_name=student-scratch --network=studentnet
```

### Effect of loss function
* Similar performance.
```
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=cse # 84.99%
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=mse # 84.85%
```

### Effect of Alpha
* alpha = 0.5 may show better performance.
```
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=cse # 84.99%
python -m trainer --T=1.0 --alpha=0.5 --kd_mode=cse # 85.38%
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=mse # 84.85%
python -m trainer --T=1.0 --alpha=0.5 --kd_mode=mse # 84.92%
```

### Effect of Temperature Scaling
* Higher the temperature, better the performance. Consistent results with the paper.
```
python -m trainer --T=1.0  --alpha=0.5 --kd_mode=cse # 85.38%
python -m trainer --T=2.0  --alpha=0.5 --kd_mode=cse # 85.27%
python -m trainer --T=4.0  --alpha=0.5 --kd_mode=cse # 86.46%
python -m trainer --T=8.0  --alpha=0.5 --kd_mode=cse # 86.33%
python -m trainer --T=16.0 --alpha=0.5 --kd_mode=cse # 86.58%
```

### More Alpha Tuning
* alpha=0.5 seems to be local optimal.
```
python -m trainer --T=16.0 --alpha=0.1 --kd_mode=cse # 85.69%
python -m trainer --T=16.0 --alpha=0.3 --kd_mode=cse # 86.48%
python -m trainer --T=16.0 --alpha=0.5 --kd_mode=cse # 86.58%
python -m trainer --T=16.0 --alpha=0.7 --kd_mode=cse # 86.16%
python -m trainer --T=16.0 --alpha=0.9 --kd_mode=cse # 86.08%
```

### SGD Testing
```
python -m trainer --T=16.0 --alpha=0.5 --kd_mode=cse --optimizer=sgd-cifar10 # [0]
python -m pretrainer --model_name=student-scratch-sgd-cifar10 --network=studentnet --optimizer=sgd-cifar10 # [1]
```

## TODO
* [ ] fix seed.
* [ ] multi gpu handling.
* [ ] split validation set.
* [ ] experiments with 5 random seed.
* [ ] remove code redundancy.
* [ ] check the optimal T is equal to calibrated T.
* [ ] Progressbar code fix in `trainer.py`.
