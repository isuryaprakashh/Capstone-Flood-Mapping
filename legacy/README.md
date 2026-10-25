# Legacy (v0) training pipeline

The first version of the project: 512 px tiles, 5-channel masks and a single-stage training
loop, run on Google Colab. It produced the weights in `weights/` that the web interface serves
(fusion: epoch 48, val mIoU 0.178; U-Net: epoch 26, val mIoU 0.175).

The research pipeline (`training/`, `scripts/run_queue.py`) supersedes it; the v0 models are
retrained there as `unet_v0` / `siamese_v0` for comparison. This folder is kept so the earlier
weights can be reproduced.

```bash
python legacy/tile_images.py --data-root ./CP1-DATASET --output-dir ./processed_v0
python legacy/train.py --model fusion --data-root ./CP1-DATASET --mask-dir ./processed_v0/masks --epochs 50 --amp
python legacy/train.py --model unet   --data-root ./CP1-DATASET --mask-dir ./processed_v0/masks --epochs 35 --amp
python legacy/eval.py --data-root ./CP1-DATASET --mask-dir ./processed_v0/masks \
  --unet-ckpt weights/unet_net_best.pth --fusion-ckpt weights/fusion_net_best.pth --split test
```

Colab: [`SpaceNet8_Colab_Training.ipynb`](SpaceNet8_Colab_Training.ipynb)
(regenerate with `python legacy/generate_colab_notebook.py`).
