```
DATA=path/to/cifar10

python train.py --model DiT-XL/2 --batch-size 96 --data-path $DATA --num-classes 10 --sps 1 --engine evo --config evo_config.py --image_size 256 --grad-ckpt
```