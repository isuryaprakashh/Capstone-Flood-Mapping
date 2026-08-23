/content/Capstone-Flood-Mapping
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 3 (delta 2), reused 3 (delta 2), pack-reused 0 (from 0)
Unpacking objects: 100% (3/3), 327 bytes | 327.00 KiB/s, done.
From https://github.com/isuryaprakashh/Capstone-Flood-Mapping
   e14a70b..2b38df2  main       -> origin/main
Updating e14a70b..2b38df2
Fast-forward
 train.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
==========================================================
🚀 SpaceNet 8 Training Pipeline for Google Colab (T4 GPU)
==========================================================
📦 Installing required packages...
GPU Active: Tesla T4
✅ Germany_Training_Public already present. Skipping.
✅ Louisiana-East_Training_Public already present. Skipping.
✅ Louisiana-West_Test_Public already present. Skipping.
✅ Masks already present in ./processed/masks. Skipping generation.
🧠 Training Siamese Cross-Attention Network (40 epochs with OneCycleLR & AMP)...
08:10:55 [INFO] Device: cuda
08:10:55 [INFO] Creating DataLoaders...
08:10:55 [INFO] Built manifest: 801 samples from 2 dirs
08:10:55 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
08:10:58 [INFO] Created DataLoaders: train=510, val=202, test=89
08:10:58 [INFO] Building model: fusion
Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
100% 83.3M/83.3M [00:00<00:00, 190MB/s]
08:11:05 [INFO] Parameters: 48.63M (48625029 trainable)
08:11:06 [INFO] ============================================================
08:11:06 [INFO] Starting training: fusion for 40 epochs
08:11:06 [INFO] ============================================================
08:11:58 [INFO]   Epoch 0 [20/63] Loss: 1.6496
08:12:36 [INFO]   Epoch 0 [40/63] Loss: 1.6377
08:13:16 [INFO]   Epoch 0 [60/63] Loss: 1.6225
08:13:56 [INFO] Epoch   0 | Train Loss: 1.6206 | Val Loss: 1.5564 | mIoU: 0.0083 | Time: 169.4s
08:14:01 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/fusion_net_best.pth
08:14:40 [INFO]   Epoch 1 [20/63] Loss: 1.5583
08:15:19 [INFO]   Epoch 1 [40/63] Loss: 1.5434
08:15:59 [INFO]   Epoch 1 [60/63] Loss: 1.5299
08:16:37 [INFO] Epoch   1 | Train Loss: 1.5279 | Val Loss: 1.4580 | mIoU: 0.0070 | Time: 155.4s
08:17:16 [INFO]   Epoch 2 [20/63] Loss: 1.4619
08:17:54 [INFO]   Epoch 2 [40/63] Loss: 1.4470
08:18:33 [INFO]   Epoch 2 [60/63] Loss: 1.4299
08:19:09 [INFO] Epoch   2 | Train Loss: 1.4273 | Val Loss: 1.4205 | mIoU: 0.0043 | Time: 152.1s
08:19:48 [INFO]   Epoch 3 [20/63] Loss: 1.3436
08:20:24 [INFO]   Epoch 3 [40/63] Loss: 1.3229
08:21:03 [INFO]   Epoch 3 [60/63] Loss: 1.3016
08:21:37 [INFO] Epoch   3 | Train Loss: 1.2989 | Val Loss: 1.2329 | mIoU: 0.0006 | Time: 148.2s
08:22:16 [INFO]   Epoch 4 [20/63] Loss: 1.2159
08:22:52 [INFO]   Epoch 4 [40/63] Loss: 1.1990
08:23:31 [INFO]   Epoch 4 [60/63] Loss: 1.1836
08:24:06 [INFO] Epoch   4 | Train Loss: 1.1816 | Val Loss: 1.1602 | mIoU: 0.0000 | Time: 148.9s
08:24:44 [INFO]   Epoch 5 [20/63] Loss: 1.1268
08:25:21 [INFO]   Epoch 5 [40/63] Loss: 1.1184
08:25:58 [INFO]   Epoch 5 [60/63] Loss: 1.1104
08:26:34 [INFO] Epoch   5 | Train Loss: 1.1094 | Val Loss: 1.0882 | mIoU: 0.0000 | Time: 148.3s
08:27:11 [INFO]   Epoch 6 [20/63] Loss: 1.0796
08:27:49 [INFO]   Epoch 6 [40/63] Loss: 1.0769
08:28:25 [INFO]   Epoch 6 [60/63] Loss: 1.0734
08:29:02 [INFO] Epoch   6 | Train Loss: 1.0729 | Val Loss: 1.0602 | mIoU: 0.0000 | Time: 147.6s
08:29:39 [INFO]   Epoch 7 [20/63] Loss: 1.0604
08:30:17 [INFO]   Epoch 7 [40/63] Loss: 1.0581
08:30:53 [INFO]   Epoch 7 [60/63] Loss: 1.0563
08:31:27 [INFO] Epoch   7 | Train Loss: 1.0560 | Val Loss: 1.0729 | mIoU: 0.0000 | Time: 145.6s
08:32:04 [INFO]   Epoch 8 [20/63] Loss: 1.0472
08:32:41 [INFO]   Epoch 8 [40/63] Loss: 1.0485
08:33:18 [INFO]   Epoch 8 [60/63] Loss: 1.0463
08:33:51 [INFO] Epoch   8 | Train Loss: 1.0461 | Val Loss: 1.0419 | mIoU: 0.0000 | Time: 144.1s
08:34:30 [INFO]   Epoch 9 [20/63] Loss: 1.0442
08:35:06 [INFO]   Epoch 9 [40/63] Loss: 1.0441
08:35:42 [INFO]   Epoch 9 [60/63] Loss: 1.0415
08:36:18 [INFO] Epoch   9 | Train Loss: 1.0409 | Val Loss: 1.0599 | mIoU: 0.0000 | Time: 146.6s
08:36:54 [INFO]   Epoch 10 [20/63] Loss: 1.0387
08:37:32 [INFO]   Epoch 10 [40/63] Loss: 1.0374
08:38:08 [INFO]   Epoch 10 [60/63] Loss: 1.0368
08:38:44 [INFO] Epoch  10 | Train Loss: 1.0372 | Val Loss: 1.0408 | mIoU: 0.0000 | Time: 145.7s
08:39:21 [INFO]   Epoch 11 [20/63] Loss: 1.0372
08:39:59 [INFO]   Epoch 11 [40/63] Loss: 1.0386
08:40:35 [INFO]   Epoch 11 [60/63] Loss: 1.0367
08:41:09 [INFO] Epoch  11 | Train Loss: 1.0362 | Val Loss: 1.0388 | mIoU: 0.0000 | Time: 145.0s
08:41:47 [INFO]   Epoch 12 [20/63] Loss: 1.0316
08:42:22 [INFO]   Epoch 12 [40/63] Loss: 1.0327
08:43:00 [INFO]   Epoch 12 [60/63] Loss: 1.0332
08:43:34 [INFO] Epoch  12 | Train Loss: 1.0334 | Val Loss: 1.1597 | mIoU: 0.0043 | Time: 145.4s
08:43:34 [INFO] Early stopping at epoch 12 (patience=12)
08:43:34 [INFO] ============================================================
08:43:34 [INFO] Training complete. Best mIoU: 0.0083
08:43:34 [INFO] Best checkpoint: ./checkpoints/fusion_net_best.pth
08:43:34 [INFO] ============================================================
🔬 Training Baseline U-Net (25 epochs)...
08:43:41 [INFO] Device: cuda
08:43:41 [INFO] Creating DataLoaders...
08:43:41 [INFO] Built manifest: 801 samples from 2 dirs
08:43:41 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
08:43:43 [INFO] Created DataLoaders: train=510, val=202, test=89
08:43:43 [INFO] Building model: unet
08:43:45 [INFO] Parameters: 47.69M (47689733 trainable)
08:43:46 [INFO] ============================================================
08:43:46 [INFO] Starting training: unet for 25 epochs
08:43:46 [INFO] ============================================================
08:44:21 [INFO]   Epoch 0 [20/63] Loss: 1.7385
08:44:58 [INFO]   Epoch 0 [40/63] Loss: 1.7111
08:45:33 [INFO]   Epoch 0 [60/63] Loss: 1.6856
08:46:07 [INFO] Epoch   0 | Train Loss: 1.6820 | Val Loss: 1.6307 | mIoU: 0.0083 | Time: 140.8s
08:46:13 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/unet_net_best.pth
08:46:49 [INFO]   Epoch 1 [20/63] Loss: 1.5892
08:47:26 [INFO]   Epoch 1 [40/63] Loss: 1.5705
08:48:01 [INFO]   Epoch 1 [60/63] Loss: 1.5513
08:48:35 [INFO] Epoch   1 | Train Loss: 1.5482 | Val Loss: 1.4776 | mIoU: 0.0074 | Time: 142.3s
08:49:12 [INFO]   Epoch 2 [20/63] Loss: 1.4678
08:49:47 [INFO]   Epoch 2 [40/63] Loss: 1.4372
08:50:23 [INFO]   Epoch 2 [60/63] Loss: 1.4074
08:50:57 [INFO] Epoch   2 | Train Loss: 1.4033 | Val Loss: 1.4427 | mIoU: 0.0053 | Time: 142.2s
08:51:33 [INFO]   Epoch 3 [20/63] Loss: 1.2846
08:52:09 [INFO]   Epoch 3 [40/63] Loss: 1.2562
08:52:44 [INFO]   Epoch 3 [60/63] Loss: 1.2345
08:53:16 [INFO] Epoch   3 | Train Loss: 1.2317 | Val Loss: 1.2010 | mIoU: 0.0012 | Time: 139.0s
08:53:53 [INFO]   Epoch 4 [20/63] Loss: 1.1546
08:54:28 [INFO]   Epoch 4 [40/63] Loss: 1.1400
08:55:03 [INFO]   Epoch 4 [60/63] Loss: 1.1289
08:55:37 [INFO] Epoch   4 | Train Loss: 1.1275 | Val Loss: 1.1143 | mIoU: 0.0000 | Time: 140.5s
08:56:12 [INFO]   Epoch 5 [20/63] Loss: 1.0899
08:56:47 [INFO]   Epoch 5 [40/63] Loss: 1.0851
08:57:24 [INFO]   Epoch 5 [60/63] Loss: 1.0812
08:57:55 [INFO] Epoch   5 | Train Loss: 1.0808 | Val Loss: 1.0673 | mIoU: 0.0000 | Time: 138.4s
08:58:30 [INFO]   Epoch 6 [20/63] Loss: 1.0642
08:59:06 [INFO]   Epoch 6 [40/63] Loss: 1.0631
08:59:40 [INFO]   Epoch 6 [60/63] Loss: 1.0600
09:00:13 [INFO] Epoch   6 | Train Loss: 1.0600 | Val Loss: 1.0588 | mIoU: 0.0000 | Time: 138.2s
09:00:48 [INFO]   Epoch 7 [20/63] Loss: 1.0525
09:01:24 [INFO]   Epoch 7 [40/63] Loss: 1.0504
09:01:59 [INFO]   Epoch 7 [60/63] Loss: 1.0497
09:02:31 [INFO] Epoch   7 | Train Loss: 1.0495 | Val Loss: 1.0500 | mIoU: 0.0000 | Time: 137.8s
09:03:08 [INFO]   Epoch 8 [20/63] Loss: 1.0457
09:03:42 [INFO]   Epoch 8 [40/63] Loss: 1.0430
09:04:17 [INFO]   Epoch 8 [60/63] Loss: 1.0443
09:04:51 [INFO] Epoch   8 | Train Loss: 1.0441 | Val Loss: 1.0449 | mIoU: 0.0000 | Time: 140.1s
09:04:51 [INFO] Early stopping at epoch 8 (patience=8)
09:04:51 [INFO] ============================================================
09:04:51 [INFO] Training complete. Best mIoU: 0.0083
09:04:51 [INFO] Best checkpoint: ./checkpoints/unet_net_best.pth
09:04:51 [INFO] ============================================================
📊 Evaluating models with TTA & SpaceNet 8 Overlap Heuristics...
09:04:56 [INFO] Device: cuda | TTA: True | Overlap Heuristics: True
09:04:56 [INFO] Built manifest: 801 samples from 2 dirs
09:04:56 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
09:04:57 [INFO] Created DataLoaders: train=510, val=202, test=89
09:04:57 [INFO] Evaluating on val set: 202 samples
09:04:57 [INFO] 
--- Evaluating U-Net Baseline ---
09:05:00 [INFO] Loaded unet from ./checkpoints/unet_net_best.pth (epoch 0, best_iou=0.008280872021930251)
09:05:08 [INFO]   [U-Net] Evaluated 10/51 batches
09:05:12 [INFO]   [U-Net] Evaluated 20/51 batches
09:05:18 [INFO]   [U-Net] Evaluated 30/51 batches
09:05:25 [INFO]   [U-Net] Evaluated 40/51 batches
09:05:30 [INFO]   [U-Net] Evaluated 50/51 batches
09:05:30 [INFO] 
--- Evaluating Siamese Fusion ---
09:05:33 [INFO] Loaded fusion from ./checkpoints/fusion_net_best.pth (epoch 0, best_iou=0.008319955953828641)
09:05:42 [INFO]   [Fusion] Evaluated 10/51 batches
09:05:49 [INFO]   [Fusion] Evaluated 20/51 batches
09:05:56 [INFO]   [Fusion] Evaluated 30/51 batches
09:06:03 [INFO]   [Fusion] Evaluated 40/51 batches
09:06:10 [INFO]   [Fusion] Evaluated 50/51 batches

==========================================================================================
MODEL COMPARISON — SpaceNet 8 Flood Mapping
==========================================================================================
Metric                   |        UNet Baseline |       Siamese Fusion | {'Δ (Fusion - UNet)':>20}
------------------------------------------------------------------------------------------
iou_building             |              0.0153 |              0.0153 | +            0.0001 
f1_building              |              0.0294 |              0.0295 | +            0.0001 
------------------------------------------------------------------------------------------
iou_road                 |              0.0124 |              0.0124 |            -0.0000 
f1_road                  |              0.0245 |              0.0245 |            -0.0000 
------------------------------------------------------------------------------------------
iou_flood                |              0.0065 |              0.0065 | +            0.0000 
f1_flood                 |              0.0126 |              0.0126 | +            0.0000 
------------------------------------------------------------------------------------------
iou_flooded_building     |              0.0042 |              0.0042 |            -0.0000 
f1_flooded_building      |              0.0082 |              0.0081 |            -0.0001 
------------------------------------------------------------------------------------------
iou_flooded_road         |              0.0023 |              0.0023 | +            0.0000 
f1_flooded_road          |              0.0046 |              0.0046 | +            0.0000 
------------------------------------------------------------------------------------------
★ mean_iou               |              0.0081 |              0.0082 | +            0.0000 
★ mean_f1                |              0.0159 |              0.0159 | +            0.0000 
==========================================================================================
09:06:10 [INFO] Results saved to ./logs/final_eval_benchmark.json
💾 Backing up trained checkpoints & benchmarks to Google Drive...
✅ Successfully backed up to /content/drive/MyDrive/SpaceNet8_Checkpoints/
==========================================================
🎉 Colab Training & Benchmark Complete!
Checkpoints: ./checkpoints/
Logs:        ./logs/
==========================================================