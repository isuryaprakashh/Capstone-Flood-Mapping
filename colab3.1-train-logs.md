Mounted at /content/drive
Cloning into '/content/Capstone-Flood-Mapping'...
remote: Enumerating objects: 185, done.
remote: Counting objects: 100% (185/185), done.
remote: Compressing objects: 100% (133/133), done.
remote: Total 185 (delta 83), reused 149 (delta 47), pack-reused 0 (from 0)
Receiving objects: 100% (185/185), 148.41 KiB | 1.14 MiB/s, done.
Resolving deltas: 100% (83/83), done.
/content/Capstone-Flood-Mapping
==========================================================
🚀 SpaceNet 8 Training Pipeline for Google Colab (T4 GPU)
==========================================================
📦 Installing required packages...
GPU Active: Tesla T4
⬇️ Downloading Germany_Training_Public from AWS S3...
CP1-DATASET/Germany 100%[===================>]   1.26G  42.7MB/s    in 31s     
📦 Extracting Germany_Training_Public...
✅ Germany_Training_Public extracted!
⬇️ Downloading Louisiana-East_Training_Public from AWS S3...
CP1-DATASET/Louisia 100%[===================>]   3.11G  38.8MB/s    in 82s     
📦 Extracting Louisiana-East_Training_Public...
✅ Louisiana-East_Training_Public extracted!
⬇️ Downloading Louisiana-West_Test_Public from AWS S3...
CP1-DATASET/Louisia 100%[===================>]   2.44G  26.4MB/s    in 68s     
📦 Extracting Louisiana-West_Test_Public...
✅ Louisiana-West_Test_Public extracted!
🗺️ Generating 5-channel segmentation masks...
2026-08-23 11:36:24 [INFO] __main__: Found dataset: Germany_Training_Public (at CP1-DATASET/Germany_Training_Public)
2026-08-23 11:36:24 [INFO] __main__: Found dataset: Louisiana-East_Training_Public (at CP1-DATASET/Louisiana-East_Training_Public)
2026-08-23 11:36:24 [INFO] __main__: Found dataset: Louisiana-West_Test_Public (at CP1-DATASET/Louisiana-West_Test_Public)
2026-08-23 11:36:24 [INFO] __main__: ============================================================
2026-08-23 11:36:24 [INFO] __main__: STEP 1: Generating segmentation masks
2026-08-23 11:36:24 [INFO] __main__: ============================================================
2026-08-23 11:36:24 [INFO] __main__:   Processing Germany_Training_Public...
2026-08-23 11:36:31 [INFO] data.mask_generator:   [Germany_Training_Public] Progress: 50/202 masks processed
2026-08-23 11:36:36 [INFO] data.mask_generator:   [Germany_Training_Public] Progress: 100/202 masks processed
2026-08-23 11:36:42 [INFO] data.mask_generator:   [Germany_Training_Public] Progress: 150/202 masks processed
2026-08-23 11:36:48 [INFO] data.mask_generator:   [Germany_Training_Public] Progress: 200/202 masks processed
2026-08-23 11:36:48 [INFO] data.mask_generator:   [Germany_Training_Public] Progress: 202/202 masks processed
2026-08-23 11:36:48 [INFO] data.mask_generator: Generated/verified 202 masks, skipped 0 in Germany_Training_Public
2026-08-23 11:36:48 [INFO] __main__:   → Generated 202 masks
2026-08-23 11:36:48 [INFO] __main__:   Processing Louisiana-East_Training_Public...
2026-08-23 11:36:57 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 50/599 masks processed
2026-08-23 11:37:05 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 100/599 masks processed
2026-08-23 11:37:14 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 150/599 masks processed
2026-08-23 11:37:22 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 200/599 masks processed
2026-08-23 11:37:31 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 250/599 masks processed
2026-08-23 11:37:40 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 300/599 masks processed
2026-08-23 11:37:48 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 350/599 masks processed
2026-08-23 11:37:57 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 400/599 masks processed
2026-08-23 11:38:05 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 450/599 masks processed
2026-08-23 11:38:14 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 500/599 masks processed
2026-08-23 11:38:23 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 550/599 masks processed
2026-08-23 11:38:30 [INFO] data.mask_generator:   [Louisiana-East_Training_Public] Progress: 599/599 masks processed
2026-08-23 11:38:30 [INFO] data.mask_generator: Generated/verified 599 masks, skipped 0 in Louisiana-East_Training_Public
2026-08-23 11:38:30 [INFO] __main__:   → Generated 599 masks
2026-08-23 11:38:30 [INFO] __main__:   Skipping Louisiana-West_Test_Public (no annotations — test set)
2026-08-23 11:38:30 [INFO] __main__: Total masks generated: 801
2026-08-23 11:38:30 [INFO] __main__: ============================================================
2026-08-23 11:38:30 [INFO] __main__: STEP 2: Building data manifest & splits
2026-08-23 11:38:30 [INFO] __main__: ============================================================
2026-08-23 11:38:30 [INFO] data.dataset: Built manifest: 801 samples from 2 dirs
2026-08-23 11:38:30 [INFO] data.dataset:   Mask lookup: 801 found, 0 missing (100% coverage)
2026-08-23 11:38:30 [INFO] __main__: Total samples in manifest: 801
2026-08-23 11:38:30 [INFO] data.dataset: Dataset split: train=510, val=202, test=89
2026-08-23 11:38:30 [INFO] __main__:   Saved train manifest: 510 samples → processed/manifests/train_manifest.json
2026-08-23 11:38:30 [INFO] __main__:   Saved val manifest: 202 samples → processed/manifests/val_manifest.json
2026-08-23 11:38:30 [INFO] __main__:   Saved test manifest: 89 samples → processed/manifests/test_manifest.json
2026-08-23 11:38:30 [INFO] __main__: ============================================================
2026-08-23 11:38:30 [INFO] __main__: PREPROCESSING COMPLETE
2026-08-23 11:38:30 [INFO] __main__: ============================================================
2026-08-23 11:38:30 [INFO] __main__:   Masks directory:    processed/masks
2026-08-23 11:38:30 [INFO] __main__:   Manifests directory: processed/manifests
2026-08-23 11:38:30 [INFO] __main__:   Train samples: 510
2026-08-23 11:38:30 [INFO] __main__:   Val samples:   202
2026-08-23 11:38:30 [INFO] __main__:   Test samples:  89
2026-08-23 11:38:30 [INFO] __main__: 
2026-08-23 11:38:30 [INFO] __main__: Next step: Run training with:
2026-08-23 11:38:30 [INFO] __main__:   python train.py --data-root CP1-DATASET --mask-dir processed/masks
🧠 Training Siamese Cross-Attention Network (50 epochs with OneCycleLR & AMP)...
11:38:35 [INFO] Device: cuda
11:38:35 [INFO] Creating DataLoaders...
11:38:36 [INFO] Built manifest: 801 samples from 2 dirs
11:38:36 [INFO]   Mask lookup: 801 found, 0 missing (100% coverage)
11:38:36 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
11:38:37 [INFO] Created DataLoaders: train=510, val=202, test=89
11:38:37 [INFO] Running data sanity check...
11:38:53 [INFO]   Sanity check: 16/25 class-batch slots have positive pixels (checked 40 samples)
11:38:53 [INFO] Building model: fusion
Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
100% 83.3M/83.3M [00:00<00:00, 181MB/s]
11:38:59 [INFO] Parameters: 48.63M (48625029 trainable)
11:38:59 [INFO] ============================================================
11:38:59 [INFO] Starting training: fusion for 50 epochs
11:38:59 [INFO] ============================================================
11:39:52 [INFO]   Epoch 0 [20/63] Loss: 2.6569
11:40:32 [INFO]   Epoch 0 [40/63] Loss: 2.6550
11:41:12 [INFO]   Epoch 0 [60/63] Loss: 2.6503
11:41:54 [INFO] Epoch   0 | Train Loss: 2.6498 | Val Loss: 2.5858 | mIoU: 0.0093 | Time: 174.0s
11:41:57 [INFO]   ★ New best model saved: mIoU=0.0093 → ./checkpoints/fusion_net_best.pth
11:42:39 [INFO]   Epoch 1 [20/63] Loss: 2.6315
11:43:19 [INFO]   Epoch 1 [40/63] Loss: 2.6264
11:43:58 [INFO]   Epoch 1 [60/63] Loss: 2.6201
11:44:38 [INFO] Epoch   1 | Train Loss: 2.6191 | Val Loss: 2.5777 | mIoU: 0.0099 | Time: 161.3s
11:44:40 [INFO]   ★ New best model saved: mIoU=0.0099 → ./checkpoints/fusion_net_best.pth
11:45:21 [INFO]   Epoch 2 [20/63] Loss: 2.5942
11:46:01 [INFO]   Epoch 2 [40/63] Loss: 2.5871
11:46:38 [INFO]   Epoch 2 [60/63] Loss: 2.5762
11:47:17 [INFO] Epoch   2 | Train Loss: 2.5751 | Val Loss: 2.5394 | mIoU: 0.0112 | Time: 157.3s
11:47:19 [INFO]   ★ New best model saved: mIoU=0.0112 → ./checkpoints/fusion_net_best.pth
11:48:00 [INFO]   Epoch 3 [20/63] Loss: 2.5347
11:48:37 [INFO]   Epoch 3 [40/63] Loss: 2.5308
11:49:15 [INFO]   Epoch 3 [60/63] Loss: 2.5234
11:49:54 [INFO] Epoch   3 | Train Loss: 2.5225 | Val Loss: 2.5035 | mIoU: 0.0133 | Time: 154.8s
11:49:56 [INFO]   ★ New best model saved: mIoU=0.0133 → ./checkpoints/fusion_net_best.pth
11:50:33 [INFO]   Epoch 4 [20/63] Loss: 2.4981
11:51:13 [INFO]   Epoch 4 [40/63] Loss: 2.4935
11:51:51 [INFO]   Epoch 4 [60/63] Loss: 2.4887
11:52:28 [INFO] Epoch   4 | Train Loss: 2.4881 | Val Loss: 2.4789 | mIoU: 0.0150 | Time: 152.5s
11:52:32 [INFO]   ★ New best model saved: mIoU=0.0150 → ./checkpoints/fusion_net_best.pth
11:53:11 [INFO]   Epoch 5 [20/63] Loss: 2.4686
11:53:48 [INFO]   Epoch 5 [40/63] Loss: 2.4677
11:54:27 [INFO]   Epoch 5 [60/63] Loss: 2.4617
11:55:05 [INFO] Epoch   5 | Train Loss: 2.4606 | Val Loss: 2.4528 | mIoU: 0.0196 | Time: 153.2s
11:55:07 [INFO]   ★ New best model saved: mIoU=0.0196 → ./checkpoints/fusion_net_best.pth
11:55:46 [INFO]   Epoch 6 [20/63] Loss: 2.4363
11:56:24 [INFO]   Epoch 6 [40/63] Loss: 2.4356
11:57:00 [INFO]   Epoch 6 [60/63] Loss: 2.4293
11:57:38 [INFO] Epoch   6 | Train Loss: 2.4285 | Val Loss: 2.4229 | mIoU: 0.0321 | Time: 151.2s
11:57:40 [INFO]   ★ New best model saved: mIoU=0.0321 → ./checkpoints/fusion_net_best.pth
11:58:19 [INFO]   Epoch 7 [20/63] Loss: 2.4036
11:58:56 [INFO]   Epoch 7 [40/63] Loss: 2.4002
11:59:33 [INFO]   Epoch 7 [60/63] Loss: 2.3950
12:00:10 [INFO] Epoch   7 | Train Loss: 2.3937 | Val Loss: 2.3816 | mIoU: 0.0381 | Time: 150.5s
12:00:12 [INFO]   ★ New best model saved: mIoU=0.0381 → ./checkpoints/fusion_net_best.pth
12:00:52 [INFO]   Epoch 8 [20/63] Loss: 2.3705
12:01:29 [INFO]   Epoch 8 [40/63] Loss: 2.3637
12:02:07 [INFO]   Epoch 8 [60/63] Loss: 2.3571
12:02:42 [INFO] Epoch   8 | Train Loss: 2.3558 | Val Loss: 2.3500 | mIoU: 0.0425 | Time: 150.0s
12:02:43 [INFO]   ★ New best model saved: mIoU=0.0425 → ./checkpoints/fusion_net_best.pth
12:03:23 [INFO]   Epoch 9 [20/63] Loss: 2.3275
12:03:59 [INFO]   Epoch 9 [40/63] Loss: 2.3215
12:04:37 [INFO]   Epoch 9 [60/63] Loss: 2.3166
12:05:12 [INFO] Epoch   9 | Train Loss: 2.3156 | Val Loss: 2.3084 | mIoU: 0.0506 | Time: 148.4s
12:05:16 [INFO]   ★ New best model saved: mIoU=0.0506 → ./checkpoints/fusion_net_best.pth
12:05:53 [INFO]   Epoch 10 [20/63] Loss: 2.2940
12:06:30 [INFO]   Epoch 10 [40/63] Loss: 2.2855
12:07:06 [INFO]   Epoch 10 [60/63] Loss: 2.2796
12:07:41 [INFO] Epoch  10 | Train Loss: 2.2785 | Val Loss: 2.2671 | mIoU: 0.0498 | Time: 145.5s
12:08:20 [INFO]   Epoch 11 [20/63] Loss: 2.2519
12:08:56 [INFO]   Epoch 11 [40/63] Loss: 2.2480
12:09:34 [INFO]   Epoch 11 [60/63] Loss: 2.2398
12:10:09 [INFO] Epoch  11 | Train Loss: 2.2388 | Val Loss: 2.2369 | mIoU: 0.0702 | Time: 147.4s
12:10:10 [INFO]   ★ New best model saved: mIoU=0.0702 → ./checkpoints/fusion_net_best.pth
12:10:50 [INFO]   Epoch 12 [20/63] Loss: 2.2171
12:11:29 [INFO]   Epoch 12 [40/63] Loss: 2.2115
12:12:06 [INFO]   Epoch 12 [60/63] Loss: 2.2047
12:12:43 [INFO] Epoch  12 | Train Loss: 2.2047 | Val Loss: 2.2028 | mIoU: 0.0751 | Time: 152.5s
12:12:46 [INFO]   ★ New best model saved: mIoU=0.0751 → ./checkpoints/fusion_net_best.pth
12:13:26 [INFO]   Epoch 13 [20/63] Loss: 2.1864
12:14:04 [INFO]   Epoch 13 [40/63] Loss: 2.1819
12:14:43 [INFO]   Epoch 13 [60/63] Loss: 2.1739
12:15:20 [INFO] Epoch  13 | Train Loss: 2.1736 | Val Loss: 2.1906 | mIoU: 0.0682 | Time: 153.5s
12:16:01 [INFO]   Epoch 14 [20/63] Loss: 2.1571
12:16:37 [INFO]   Epoch 14 [40/63] Loss: 2.1523
12:17:14 [INFO]   Epoch 14 [60/63] Loss: 2.1492
12:17:50 [INFO] Epoch  14 | Train Loss: 2.1493 | Val Loss: 2.1677 | mIoU: 0.0798 | Time: 150.5s
12:17:57 [INFO]   ★ New best model saved: mIoU=0.0798 → ./checkpoints/fusion_net_best.pth
12:18:37 [INFO]   Epoch 15 [20/63] Loss: 2.1276
12:19:17 [INFO]   Epoch 15 [40/63] Loss: 2.1298
12:19:56 [INFO]   Epoch 15 [60/63] Loss: 2.1290
12:20:34 [INFO] Epoch  15 | Train Loss: 2.1285 | Val Loss: 2.1528 | mIoU: 0.0891 | Time: 157.1s
12:20:36 [INFO]   ★ New best model saved: mIoU=0.0891 → ./checkpoints/fusion_net_best.pth
12:21:18 [INFO]   Epoch 16 [20/63] Loss: 2.1131
12:21:58 [INFO]   Epoch 16 [40/63] Loss: 2.1171
12:22:35 [INFO]   Epoch 16 [60/63] Loss: 2.1122
12:23:11 [INFO] Epoch  16 | Train Loss: 2.1114 | Val Loss: 2.1460 | mIoU: 0.0822 | Time: 154.6s
12:23:52 [INFO]   Epoch 17 [20/63] Loss: 2.0930
12:24:30 [INFO]   Epoch 17 [40/63] Loss: 2.0957
12:25:06 [INFO]   Epoch 17 [60/63] Loss: 2.0886
12:25:44 [INFO] Epoch  17 | Train Loss: 2.0889 | Val Loss: 2.1280 | mIoU: 0.1456 | Time: 153.1s
12:25:51 [INFO]   ★ New best model saved: mIoU=0.1456 → ./checkpoints/fusion_net_best.pth
12:26:32 [INFO]   Epoch 18 [20/63] Loss: 2.0805
12:27:08 [INFO]   Epoch 18 [40/63] Loss: 2.0761
12:27:46 [INFO]   Epoch 18 [60/63] Loss: 2.0708
12:28:22 [INFO] Epoch  18 | Train Loss: 2.0713 | Val Loss: 2.1192 | mIoU: 0.1365 | Time: 150.6s
12:29:02 [INFO]   Epoch 19 [20/63] Loss: 2.0602
12:29:39 [INFO]   Epoch 19 [40/63] Loss: 2.0619
12:30:17 [INFO]   Epoch 19 [60/63] Loss: 2.0574
12:30:52 [INFO] Epoch  19 | Train Loss: 2.0576 | Val Loss: 2.1158 | mIoU: 0.1197 | Time: 150.6s
12:31:33 [INFO]   Epoch 20 [20/63] Loss: 2.0439
12:32:09 [INFO]   Epoch 20 [40/63] Loss: 2.0465
12:32:48 [INFO]   Epoch 20 [60/63] Loss: 2.0442
12:33:25 [INFO] Epoch  20 | Train Loss: 2.0445 | Val Loss: 2.1072 | mIoU: 0.1670 | Time: 152.3s
12:33:32 [INFO]   ★ New best model saved: mIoU=0.1670 → ./checkpoints/fusion_net_best.pth
12:34:13 [INFO]   Epoch 21 [20/63] Loss: 2.0359
12:34:51 [INFO]   Epoch 21 [40/63] Loss: 2.0373
12:35:29 [INFO]   Epoch 21 [60/63] Loss: 2.0292
12:36:03 [INFO] Epoch  21 | Train Loss: 2.0286 | Val Loss: 2.1160 | mIoU: 0.1108 | Time: 151.2s
12:36:42 [INFO]   Epoch 22 [20/63] Loss: 2.0296
12:37:17 [INFO]   Epoch 22 [40/63] Loss: 2.0246
12:37:53 [INFO]   Epoch 22 [60/63] Loss: 2.0196
12:38:29 [INFO] Epoch  22 | Train Loss: 2.0208 | Val Loss: 2.0968 | mIoU: 0.1133 | Time: 145.8s
12:39:07 [INFO]   Epoch 23 [20/63] Loss: 2.0157
12:39:44 [INFO]   Epoch 23 [40/63] Loss: 2.0170
12:40:19 [INFO]   Epoch 23 [60/63] Loss: 2.0138
12:40:56 [INFO] Epoch  23 | Train Loss: 2.0135 | Val Loss: 2.1120 | mIoU: 0.1096 | Time: 147.1s
12:41:33 [INFO]   Epoch 24 [20/63] Loss: 2.0063
12:42:11 [INFO]   Epoch 24 [40/63] Loss: 2.0040
12:42:47 [INFO]   Epoch 24 [60/63] Loss: 2.0058
12:43:24 [INFO] Epoch  24 | Train Loss: 2.0059 | Val Loss: 2.0848 | mIoU: 0.1618 | Time: 147.3s
12:44:01 [INFO]   Epoch 25 [20/63] Loss: 1.9954
12:44:39 [INFO]   Epoch 25 [40/63] Loss: 1.9954
12:45:15 [INFO]   Epoch 25 [60/63] Loss: 1.9937
12:45:51 [INFO] Epoch  25 | Train Loss: 1.9955 | Val Loss: 2.0654 | mIoU: 0.1706 | Time: 147.2s
12:46:03 [INFO]   ★ New best model saved: mIoU=0.1706 → ./checkpoints/fusion_net_best.pth
12:46:41 [INFO]   Epoch 26 [20/63] Loss: 2.0276
12:47:19 [INFO]   Epoch 26 [40/63] Loss: 2.0085
12:47:53 [INFO]   Epoch 26 [60/63] Loss: 1.9934
12:48:28 [INFO] Epoch  26 | Train Loss: 1.9924 | Val Loss: 2.0853 | mIoU: 0.1480 | Time: 144.7s
12:49:06 [INFO]   Epoch 27 [20/63] Loss: 1.9808
12:49:41 [INFO]   Epoch 27 [40/63] Loss: 1.9921
12:50:19 [INFO]   Epoch 27 [60/63] Loss: 1.9839
12:50:54 [INFO] Epoch  27 | Train Loss: 1.9844 | Val Loss: 2.0829 | mIoU: 0.1745 | Time: 146.8s
12:50:58 [INFO]   ★ New best model saved: mIoU=0.1745 → ./checkpoints/fusion_net_best.pth
12:51:36 [INFO]   Epoch 28 [20/63] Loss: 1.9739
12:52:14 [INFO]   Epoch 28 [40/63] Loss: 1.9744
12:52:49 [INFO]   Epoch 28 [60/63] Loss: 1.9768
12:53:24 [INFO] Epoch  28 | Train Loss: 1.9762 | Val Loss: 2.0770 | mIoU: 0.1580 | Time: 145.6s
12:54:02 [INFO]   Epoch 29 [20/63] Loss: 1.9495
12:54:37 [INFO]   Epoch 29 [40/63] Loss: 1.9685
12:55:15 [INFO]   Epoch 29 [60/63] Loss: 1.9715
12:55:49 [INFO] Epoch  29 | Train Loss: 1.9726 | Val Loss: 2.0780 | mIoU: 0.1622 | Time: 145.1s
12:56:28 [INFO]   Epoch 30 [20/63] Loss: 1.9632
12:57:03 [INFO]   Epoch 30 [40/63] Loss: 1.9760
12:57:40 [INFO]   Epoch 30 [60/63] Loss: 1.9699
12:58:15 [INFO] Epoch  30 | Train Loss: 1.9686 | Val Loss: 2.0773 | mIoU: 0.1626 | Time: 145.9s
12:58:52 [INFO]   Epoch 31 [20/63] Loss: 1.9485
12:59:29 [INFO]   Epoch 31 [40/63] Loss: 1.9582
13:00:04 [INFO]   Epoch 31 [60/63] Loss: 1.9612
13:00:40 [INFO] Epoch  31 | Train Loss: 1.9631 | Val Loss: 2.0631 | mIoU: 0.1580 | Time: 145.4s
13:01:16 [INFO]   Epoch 32 [20/63] Loss: 1.9808
13:01:53 [INFO]   Epoch 32 [40/63] Loss: 1.9611
13:02:29 [INFO]   Epoch 32 [60/63] Loss: 1.9569
13:03:03 [INFO] Epoch  32 | Train Loss: 1.9583 | Val Loss: 2.0712 | mIoU: 0.1685 | Time: 143.2s
13:03:41 [INFO]   Epoch 33 [20/63] Loss: 1.9741
13:04:16 [INFO]   Epoch 33 [40/63] Loss: 1.9671
13:04:52 [INFO]   Epoch 33 [60/63] Loss: 1.9583
13:05:27 [INFO] Epoch  33 | Train Loss: 1.9579 | Val Loss: 2.0732 | mIoU: 0.1683 | Time: 144.0s
13:06:06 [INFO]   Epoch 34 [20/63] Loss: 1.9352
13:06:43 [INFO]   Epoch 34 [40/63] Loss: 1.9478
13:07:21 [INFO]   Epoch 34 [60/63] Loss: 1.9477
13:07:56 [INFO] Epoch  34 | Train Loss: 1.9478 | Val Loss: 2.0752 | mIoU: 0.1561 | Time: 149.1s
13:08:36 [INFO]   Epoch 35 [20/63] Loss: 1.9458
13:09:12 [INFO]   Epoch 35 [40/63] Loss: 1.9419
13:09:49 [INFO]   Epoch 35 [60/63] Loss: 1.9455
13:10:24 [INFO] Epoch  35 | Train Loss: 1.9461 | Val Loss: 2.0710 | mIoU: 0.1699 | Time: 147.7s
13:11:02 [INFO]   Epoch 36 [20/63] Loss: 1.9542
13:11:37 [INFO]   Epoch 36 [40/63] Loss: 1.9458
13:12:12 [INFO]   Epoch 36 [60/63] Loss: 1.9413
13:12:49 [INFO] Epoch  36 | Train Loss: 1.9432 | Val Loss: 2.0760 | mIoU: 0.1665 | Time: 144.7s
13:13:26 [INFO]   Epoch 37 [20/63] Loss: 1.9287
13:14:03 [INFO]   Epoch 37 [40/63] Loss: 1.9382
13:14:38 [INFO]   Epoch 37 [60/63] Loss: 1.9375
13:15:14 [INFO] Epoch  37 | Train Loss: 1.9388 | Val Loss: 2.0759 | mIoU: 0.1697 | Time: 145.6s
13:15:52 [INFO]   Epoch 38 [20/63] Loss: 1.9522
13:16:29 [INFO]   Epoch 38 [40/63] Loss: 1.9451
13:17:05 [INFO]   Epoch 38 [60/63] Loss: 1.9406
13:17:39 [INFO] Epoch  38 | Train Loss: 1.9412 | Val Loss: 2.0725 | mIoU: 0.1651 | Time: 144.4s
13:18:17 [INFO]   Epoch 39 [20/63] Loss: 1.9416
13:18:54 [INFO]   Epoch 39 [40/63] Loss: 1.9455
13:19:32 [INFO]   Epoch 39 [60/63] Loss: 1.9415
13:20:07 [INFO] Epoch  39 | Train Loss: 1.9436 | Val Loss: 2.0819 | mIoU: 0.1659 | Time: 147.8s
13:20:46 [INFO]   Epoch 40 [20/63] Loss: 1.9638
13:21:22 [INFO]   Epoch 40 [40/63] Loss: 1.9474
13:21:59 [INFO]   Epoch 40 [60/63] Loss: 1.9360
13:22:34 [INFO] Epoch  40 | Train Loss: 1.9333 | Val Loss: 2.0728 | mIoU: 0.1723 | Time: 147.2s
13:23:12 [INFO]   Epoch 41 [20/63] Loss: 1.9394
13:23:48 [INFO]   Epoch 41 [40/63] Loss: 1.9345
13:24:26 [INFO]   Epoch 41 [60/63] Loss: 1.9333
13:25:00 [INFO] Epoch  41 | Train Loss: 1.9335 | Val Loss: 2.0724 | mIoU: 0.1735 | Time: 146.2s
13:25:38 [INFO]   Epoch 42 [20/63] Loss: 1.9203
13:26:14 [INFO]   Epoch 42 [40/63] Loss: 1.9245
13:26:50 [INFO]   Epoch 42 [60/63] Loss: 1.9270
13:27:26 [INFO] Epoch  42 | Train Loss: 1.9267 | Val Loss: 2.0743 | mIoU: 0.1747 | Time: 146.2s
13:27:29 [INFO]   ★ New best model saved: mIoU=0.1747 → ./checkpoints/fusion_net_best.pth
13:28:09 [INFO]   Epoch 43 [20/63] Loss: 1.9213
13:28:45 [INFO]   Epoch 43 [40/63] Loss: 1.9249
13:29:22 [INFO]   Epoch 43 [60/63] Loss: 1.9256
13:29:57 [INFO] Epoch  43 | Train Loss: 1.9251 | Val Loss: 2.0693 | mIoU: 0.1728 | Time: 147.5s
13:30:35 [INFO]   Epoch 44 [20/63] Loss: 1.9176
13:31:11 [INFO]   Epoch 44 [40/63] Loss: 1.9244
13:31:46 [INFO]   Epoch 44 [60/63] Loss: 1.9276
13:32:22 [INFO] Epoch  44 | Train Loss: 1.9264 | Val Loss: 2.0649 | mIoU: 0.1759 | Time: 145.5s
13:32:25 [INFO]   ★ New best model saved: mIoU=0.1759 → ./checkpoints/fusion_net_best.pth
13:33:02 [INFO]   Epoch 45 [20/63] Loss: 1.9412
13:33:39 [INFO]   Epoch 45 [40/63] Loss: 1.9323
13:34:14 [INFO]   Epoch 45 [60/63] Loss: 1.9282
13:34:51 [INFO] Epoch  45 | Train Loss: 1.9272 | Val Loss: 2.0735 | mIoU: 0.1732 | Time: 146.6s
13:35:28 [INFO]   Epoch 46 [20/63] Loss: 1.9252
13:36:05 [INFO]   Epoch 46 [40/63] Loss: 1.9303
13:36:40 [INFO]   Epoch 46 [60/63] Loss: 1.9307
13:37:16 [INFO] Epoch  46 | Train Loss: 1.9310 | Val Loss: 2.0657 | mIoU: 0.1753 | Time: 144.8s
13:37:53 [INFO]   Epoch 47 [20/63] Loss: 1.9356
13:38:29 [INFO]   Epoch 47 [40/63] Loss: 1.9272
13:39:04 [INFO]   Epoch 47 [60/63] Loss: 1.9247
13:39:40 [INFO] Epoch  47 | Train Loss: 1.9260 | Val Loss: 2.0611 | mIoU: 0.1747 | Time: 144.3s
13:40:19 [INFO]   Epoch 48 [20/63] Loss: 1.9399
13:40:55 [INFO]   Epoch 48 [40/63] Loss: 1.9368
13:41:31 [INFO]   Epoch 48 [60/63] Loss: 1.9269
13:42:07 [INFO] Epoch  48 | Train Loss: 1.9252 | Val Loss: 2.0725 | mIoU: 0.1778 | Time: 146.6s
13:42:09 [INFO]   ★ New best model saved: mIoU=0.1778 → ./checkpoints/fusion_net_best.pth
13:42:48 [INFO]   Epoch 49 [20/63] Loss: 1.9338
13:43:24 [INFO]   Epoch 49 [40/63] Loss: 1.9243
13:44:02 [INFO]   Epoch 49 [60/63] Loss: 1.9217
13:44:36 [INFO] Epoch  49 | Train Loss: 1.9230 | Val Loss: 2.0675 | mIoU: 0.1761 | Time: 147.6s
13:44:36 [INFO] ============================================================
13:44:36 [INFO] Training complete. Best mIoU: 0.1778
13:44:36 [INFO] Best checkpoint: ./checkpoints/fusion_net_best.pth
13:44:36 [INFO] ============================================================
🔬 Training Baseline U-Net (35 epochs)...
13:44:47 [INFO] Device: cuda
13:44:47 [INFO] Creating DataLoaders...
13:44:47 [INFO] Built manifest: 801 samples from 2 dirs
13:44:47 [INFO]   Mask lookup: 801 found, 0 missing (100% coverage)
13:44:47 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
13:44:49 [INFO] Created DataLoaders: train=510, val=202, test=89
13:44:49 [INFO] Running data sanity check...
13:45:04 [INFO]   Sanity check: 23/25 class-batch slots have positive pixels (checked 40 samples)
13:45:04 [INFO] Building model: unet
13:45:06 [INFO] Parameters: 47.69M (47689733 trainable)
13:45:06 [INFO] ============================================================
13:45:06 [INFO] Starting training: unet for 35 epochs
13:45:06 [INFO] ============================================================
13:45:45 [INFO]   Epoch 0 [20/63] Loss: 2.6789
13:46:20 [INFO]   Epoch 0 [40/63] Loss: 2.6742
13:46:57 [INFO]   Epoch 0 [60/63] Loss: 2.6709
13:47:32 [INFO] Epoch   0 | Train Loss: 2.6703 | Val Loss: 2.6164 | mIoU: 0.0082 | Time: 144.9s
13:47:36 [INFO]   ★ New best model saved: mIoU=0.0082 → ./checkpoints/unet_net_best.pth
13:48:12 [INFO]   Epoch 1 [20/63] Loss: 2.6484
13:48:47 [INFO]   Epoch 1 [40/63] Loss: 2.6329
13:49:22 [INFO]   Epoch 1 [60/63] Loss: 2.6199
13:49:57 [INFO] Epoch   1 | Train Loss: 2.6177 | Val Loss: 2.5582 | mIoU: 0.0088 | Time: 140.5s
13:50:04 [INFO]   ★ New best model saved: mIoU=0.0088 → ./checkpoints/unet_net_best.pth
13:50:41 [INFO]   Epoch 2 [20/63] Loss: 2.5648
13:51:15 [INFO]   Epoch 2 [40/63] Loss: 2.5536
13:51:53 [INFO]   Epoch 2 [60/63] Loss: 2.5458
13:52:30 [INFO] Epoch   2 | Train Loss: 2.5443 | Val Loss: 2.4967 | mIoU: 0.0112 | Time: 146.3s
13:52:37 [INFO]   ★ New best model saved: mIoU=0.0112 → ./checkpoints/unet_net_best.pth
13:53:15 [INFO]   Epoch 3 [20/63] Loss: 2.5174
13:53:53 [INFO]   Epoch 3 [40/63] Loss: 2.5136
13:54:30 [INFO]   Epoch 3 [60/63] Loss: 2.5086
13:55:06 [INFO] Epoch   3 | Train Loss: 2.5080 | Val Loss: 2.4632 | mIoU: 0.0141 | Time: 148.8s
13:55:08 [INFO]   ★ New best model saved: mIoU=0.0141 → ./checkpoints/unet_net_best.pth
13:55:44 [INFO]   Epoch 4 [20/63] Loss: 2.4875
13:56:22 [INFO]   Epoch 4 [40/63] Loss: 2.4850
13:56:57 [INFO]   Epoch 4 [60/63] Loss: 2.4804
13:57:32 [INFO] Epoch   4 | Train Loss: 2.4795 | Val Loss: 2.4301 | mIoU: 0.0211 | Time: 144.4s
13:57:40 [INFO]   ★ New best model saved: mIoU=0.0211 → ./checkpoints/unet_net_best.pth
13:58:19 [INFO]   Epoch 5 [20/63] Loss: 2.4647
13:58:54 [INFO]   Epoch 5 [40/63] Loss: 2.4573
13:59:28 [INFO]   Epoch 5 [60/63] Loss: 2.4510
14:00:05 [INFO] Epoch   5 | Train Loss: 2.4502 | Val Loss: 2.4034 | mIoU: 0.0297 | Time: 144.6s
14:00:07 [INFO]   ★ New best model saved: mIoU=0.0297 → ./checkpoints/unet_net_best.pth
14:00:43 [INFO]   Epoch 6 [20/63] Loss: 2.4221
14:01:20 [INFO]   Epoch 6 [40/63] Loss: 2.4172
14:01:56 [INFO]   Epoch 6 [60/63] Loss: 2.4131
14:02:30 [INFO] Epoch   6 | Train Loss: 2.4119 | Val Loss: 2.3799 | mIoU: 0.0405 | Time: 143.4s
14:02:32 [INFO]   ★ New best model saved: mIoU=0.0405 → ./checkpoints/unet_net_best.pth
14:03:08 [INFO]   Epoch 7 [20/63] Loss: 2.3880
14:03:45 [INFO]   Epoch 7 [40/63] Loss: 2.3806
14:04:21 [INFO]   Epoch 7 [60/63] Loss: 2.3731
14:04:54 [INFO] Epoch   7 | Train Loss: 2.3720 | Val Loss: 2.3257 | mIoU: 0.0536 | Time: 141.9s
14:05:01 [INFO]   ★ New best model saved: mIoU=0.0536 → ./checkpoints/unet_net_best.pth
14:05:38 [INFO]   Epoch 8 [20/63] Loss: 2.3401
14:06:16 [INFO]   Epoch 8 [40/63] Loss: 2.3360
14:06:51 [INFO]   Epoch 8 [60/63] Loss: 2.3274
14:07:25 [INFO] Epoch   8 | Train Loss: 2.3259 | Val Loss: 2.3005 | mIoU: 0.0574 | Time: 144.1s
14:07:43 [INFO]   ★ New best model saved: mIoU=0.0574 → ./checkpoints/unet_net_best.pth
14:08:19 [INFO]   Epoch 9 [20/63] Loss: 2.2916
14:08:56 [INFO]   Epoch 9 [40/63] Loss: 2.2858
14:09:31 [INFO]   Epoch 9 [60/63] Loss: 2.2795
14:10:05 [INFO] Epoch   9 | Train Loss: 2.2801 | Val Loss: 2.2577 | mIoU: 0.0668 | Time: 142.5s
14:10:28 [INFO]   ★ New best model saved: mIoU=0.0668 → ./checkpoints/unet_net_best.pth
14:11:05 [INFO]   Epoch 10 [20/63] Loss: 2.2535
14:11:40 [INFO]   Epoch 10 [40/63] Loss: 2.2426
14:12:15 [INFO]   Epoch 10 [60/63] Loss: 2.2365
14:12:52 [INFO] Epoch  10 | Train Loss: 2.2362 | Val Loss: 2.2323 | mIoU: 0.0598 | Time: 144.4s
14:13:28 [INFO]   Epoch 11 [20/63] Loss: 2.2115
14:14:06 [INFO]   Epoch 11 [40/63] Loss: 2.2070
14:14:41 [INFO]   Epoch 11 [60/63] Loss: 2.2015
14:15:17 [INFO] Epoch  11 | Train Loss: 2.2006 | Val Loss: 2.1866 | mIoU: 0.0935 | Time: 144.8s
14:15:25 [INFO]   ★ New best model saved: mIoU=0.0935 → ./checkpoints/unet_net_best.pth
14:16:04 [INFO]   Epoch 12 [20/63] Loss: 2.1819
14:16:39 [INFO]   Epoch 12 [40/63] Loss: 2.1778
14:17:14 [INFO]   Epoch 12 [60/63] Loss: 2.1736
14:17:51 [INFO] Epoch  12 | Train Loss: 2.1726 | Val Loss: 2.1695 | mIoU: 0.0874 | Time: 145.5s
14:18:26 [INFO]   Epoch 13 [20/63] Loss: 2.1556
14:19:03 [INFO]   Epoch 13 [40/63] Loss: 2.1482
14:19:39 [INFO]   Epoch 13 [60/63] Loss: 2.1479
14:20:13 [INFO] Epoch  13 | Train Loss: 2.1471 | Val Loss: 2.1575 | mIoU: 0.0790 | Time: 142.7s
14:20:51 [INFO]   Epoch 14 [20/63] Loss: 2.1330
14:21:28 [INFO]   Epoch 14 [40/63] Loss: 2.1336
14:22:05 [INFO]   Epoch 14 [60/63] Loss: 2.1292
14:22:38 [INFO] Epoch  14 | Train Loss: 2.1281 | Val Loss: 2.1542 | mIoU: 0.0589 | Time: 145.1s
14:23:17 [INFO]   Epoch 15 [20/63] Loss: 2.1124
14:23:53 [INFO]   Epoch 15 [40/63] Loss: 2.1138
14:24:30 [INFO]   Epoch 15 [60/63] Loss: 2.1108
14:25:04 [INFO] Epoch  15 | Train Loss: 2.1098 | Val Loss: 2.1351 | mIoU: 0.0891 | Time: 145.4s
14:25:39 [INFO]   Epoch 16 [20/63] Loss: 2.0951
14:26:17 [INFO]   Epoch 16 [40/63] Loss: 2.0941
14:26:52 [INFO]   Epoch 16 [60/63] Loss: 2.0941
14:27:26 [INFO] Epoch  16 | Train Loss: 2.0936 | Val Loss: 2.1177 | mIoU: 0.1119 | Time: 142.6s
14:27:33 [INFO]   ★ New best model saved: mIoU=0.1119 → ./checkpoints/unet_net_best.pth
14:28:09 [INFO]   Epoch 17 [20/63] Loss: 2.0906
14:28:46 [INFO]   Epoch 17 [40/63] Loss: 2.0833
14:29:22 [INFO]   Epoch 17 [60/63] Loss: 2.0792
14:29:57 [INFO] Epoch  17 | Train Loss: 2.0790 | Val Loss: 2.1400 | mIoU: 0.0918 | Time: 143.9s
14:30:32 [INFO]   Epoch 18 [20/63] Loss: 2.0663
14:31:10 [INFO]   Epoch 18 [40/63] Loss: 2.0684
14:31:47 [INFO]   Epoch 18 [60/63] Loss: 2.0663
14:32:22 [INFO] Epoch  18 | Train Loss: 2.0655 | Val Loss: 2.0953 | mIoU: 0.1492 | Time: 144.9s
14:32:30 [INFO]   ★ New best model saved: mIoU=0.1492 → ./checkpoints/unet_net_best.pth
14:33:07 [INFO]   Epoch 19 [20/63] Loss: 2.0554
14:33:45 [INFO]   Epoch 19 [40/63] Loss: 2.0547
14:34:20 [INFO]   Epoch 19 [60/63] Loss: 2.0548
14:34:54 [INFO] Epoch  19 | Train Loss: 2.0534 | Val Loss: 2.0936 | mIoU: 0.1495 | Time: 144.8s
14:34:57 [INFO]   ★ New best model saved: mIoU=0.1495 → ./checkpoints/unet_net_best.pth
14:35:34 [INFO]   Epoch 20 [20/63] Loss: 2.0457
14:36:12 [INFO]   Epoch 20 [40/63] Loss: 2.0406
14:36:48 [INFO]   Epoch 20 [60/63] Loss: 2.0410
14:37:20 [INFO] Epoch  20 | Train Loss: 2.0404 | Val Loss: 2.0869 | mIoU: 0.1470 | Time: 142.5s
14:37:56 [INFO]   Epoch 21 [20/63] Loss: 2.0422
14:38:31 [INFO]   Epoch 21 [40/63] Loss: 2.0355
14:39:08 [INFO]   Epoch 21 [60/63] Loss: 2.0350
14:39:41 [INFO] Epoch  21 | Train Loss: 2.0348 | Val Loss: 2.0676 | mIoU: 0.1378 | Time: 141.6s
14:40:17 [INFO]   Epoch 22 [20/63] Loss: 2.0277
14:40:54 [INFO]   Epoch 22 [40/63] Loss: 2.0221
14:41:29 [INFO]   Epoch 22 [60/63] Loss: 2.0221
14:42:02 [INFO] Epoch  22 | Train Loss: 2.0229 | Val Loss: 2.0926 | mIoU: 0.1361 | Time: 140.9s
14:42:40 [INFO]   Epoch 23 [20/63] Loss: 2.0141
14:43:16 [INFO]   Epoch 23 [40/63] Loss: 2.0177
14:43:53 [INFO]   Epoch 23 [60/63] Loss: 2.0148
14:44:26 [INFO] Epoch  23 | Train Loss: 2.0134 | Val Loss: 2.0709 | mIoU: 0.1433 | Time: 143.9s
14:45:01 [INFO]   Epoch 24 [20/63] Loss: 2.0057
14:45:37 [INFO]   Epoch 24 [40/63] Loss: 2.0082
14:46:12 [INFO]   Epoch 24 [60/63] Loss: 2.0081
14:46:45 [INFO] Epoch  24 | Train Loss: 2.0060 | Val Loss: 2.0685 | mIoU: 0.1633 | Time: 139.2s
14:46:57 [INFO]   ★ New best model saved: mIoU=0.1633 → ./checkpoints/unet_net_best.pth