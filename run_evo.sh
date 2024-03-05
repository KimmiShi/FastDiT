MODEL=3B
BS=48
numg=4
sps=2
ENGINE=evo
HW=512
VAE_PATH=/mnt/petrelfs/share_data/wangyaohui/pretrained/stable-diffusion-2-1-base/vae

MODEL=XL
HW=256
ENGINE='evo'
# ENGINE='optix'

srun --quotatype='auto' -n$numg -N1 --gres=gpu:$numg -p saturn-v python train.py --model DiT-${MODEL}/2 --batch-size $BS --data-path /mnt/petrelfs/share_data/shidongxing/torchvision-cifar10 --num-classes 10 --sps $sps --engine $ENGINE --config evo_config.py --image_size $HW --grad-ckpt --vae $VAE_PATH



# python train.py --model DiT-XL/2 --batch-size 96 --data-path /mnt/petrelfs/share_data/shidongxing/torchvision-cifar10 --num-classes 10 --sps 1 --engine evo --config evo_config.py --image_size 256 --grad-ckpt