/content/Capstone-Flood-Mapping
remote: Enumerating objects: 17, done.
remote: Counting objects: 100% (17/17), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 9 (delta 7), reused 9 (delta 7), pack-reused 0 (from 0)
Unpacking objects: 100% (9/9), 2.64 KiB | 337.00 KiB/s, done.
From https://github.com/isuryaprakashh/Capstone-Flood-Mapping
   2b38df2..82d1eae  main       -> origin/main
Updating 2b38df2..82d1eae
Fast-forward
 data/dataset.py         | 18 +++++++++++++++++-
 losses/combined_loss.py | 47 +++++++++++++++++++++++++++++++++++------------
 scripts/setup_colab.sh  | 18 ++++++++++++------
 train.py                | 28 ++++++++++++++++++++++++----
 4 files changed, 88 insertions(+), 23 deletions(-)
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
09:15:43 [INFO] Device: cuda
09:15:43 [INFO] Creating DataLoaders...
09:15:43 [INFO] Built manifest: 801 samples from 2 dirs
09:15:43 [INFO]   Mask lookup: 801 found, 0 missing (100% coverage)
09:15:43 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
09:15:44 [INFO] Created DataLoaders: train=510, val=202, test=89
09:15:44 [INFO] Running data sanity check...
09:15:51 [INFO]   Sanity check: 20/25 class-batch slots have positive pixels (checked 40 samples)
09:15:51 [INFO] Building model: fusion
09:15:54 [INFO] Parameters: 48.63M (48625029 trainable)
09:15:54 [INFO] ============================================================
09:15:54 [INFO] Starting training: fusion for 40 epochs
09:15:54 [INFO] ============================================================
09:16:25 [INFO]   Epoch 0 [20/63] Loss: 2.6300
09:16:45 [INFO]   Epoch 0 [40/63] Loss: 2.6271
09:17:05 [INFO]   Epoch 0 [60/63] Loss: 2.6228
09:17:29 [INFO] Epoch   0 | Train Loss: 2.6222 | Val Loss: 2.6113 | mIoU: 0.0083 | Time: 95.0s
09:17:33 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/fusion_net_best.pth
09:17:54 [INFO]   Epoch 1 [20/63] Loss: 2.6032
09:18:14 [INFO]   Epoch 1 [40/63] Loss: 2.6025
09:18:36 [INFO]   Epoch 1 [60/63] Loss: 2.5972
09:18:59 [INFO] Epoch   1 | Train Loss: 2.5969 | Val Loss: 2.5823 | mIoU: 0.0083 | Time: 85.7s
09:19:01 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/fusion_net_best.pth
09:19:24 [INFO]   Epoch 2 [20/63] Loss: 2.5787
09:19:43 [INFO]   Epoch 2 [40/63] Loss: 2.5766
09:20:05 [INFO]   Epoch 2 [60/63] Loss: 2.5701
09:20:27 [INFO] Epoch   2 | Train Loss: 2.5694 | Val Loss: 2.5644 | mIoU: 0.0083 | Time: 86.7s
09:20:29 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/fusion_net_best.pth
09:20:52 [INFO]   Epoch 3 [20/63] Loss: 2.5412
09:21:11 [INFO]   Epoch 3 [40/63] Loss: 2.5415
09:21:31 [INFO]   Epoch 3 [60/63] Loss: 2.5329
09:21:54 [INFO] Epoch   3 | Train Loss: 2.5321 | Val Loss: 2.5392 | mIoU: 0.0083 | Time: 84.9s
09:21:56 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/fusion_net_best.pth
09:22:18 [INFO]   Epoch 4 [20/63] Loss: 2.5061
09:22:38 [INFO]   Epoch 4 [40/63] Loss: 2.5045
09:22:57 [INFO]   Epoch 4 [60/63] Loss: 2.4964
09:23:20 [INFO] Epoch   4 | Train Loss: 2.4953 | Val Loss: 2.4890 | mIoU: 0.0075 | Time: 85.0s
09:23:41 [INFO]   Epoch 5 [20/63] Loss: 2.4659
09:24:01 [INFO]   Epoch 5 [40/63] Loss: 2.4644
09:24:21 [INFO]   Epoch 5 [60/63] Loss: 2.4558
09:24:44 [INFO] Epoch   5 | Train Loss: 2.4545 | Val Loss: 2.4592 | mIoU: 0.0071 | Time: 83.5s
09:25:06 [INFO]   Epoch 6 [20/63] Loss: 2.4218
09:25:25 [INFO]   Epoch 6 [40/63] Loss: 2.4207
09:25:47 [INFO]   Epoch 6 [60/63] Loss: 2.4122
09:26:09 [INFO] Epoch   6 | Train Loss: 2.4108 | Val Loss: 2.4032 | mIoU: 0.0061 | Time: 85.3s
09:26:30 [INFO]   Epoch 7 [20/63] Loss: 2.3766
09:26:51 [INFO]   Epoch 7 [40/63] Loss: 2.3722
09:27:10 [INFO]   Epoch 7 [60/63] Loss: 2.3630
09:27:34 [INFO] Epoch   7 | Train Loss: 2.3626 | Val Loss: 2.3676 | mIoU: 0.0048 | Time: 84.2s
09:27:54 [INFO]   Epoch 8 [20/63] Loss: 2.3350
09:28:15 [INFO]   Epoch 8 [40/63] Loss: 2.3259
09:28:35 [INFO]   Epoch 8 [60/63] Loss: 2.3172
09:28:58 [INFO] Epoch   8 | Train Loss: 2.3157 | Val Loss: 2.3418 | mIoU: 0.0031 | Time: 84.1s
09:29:20 [INFO]   Epoch 9 [20/63] Loss: 2.2834
09:29:39 [INFO]   Epoch 9 [40/63] Loss: 2.2754
09:29:59 [INFO]   Epoch 9 [60/63] Loss: 2.2748
09:30:23 [INFO] Epoch   9 | Train Loss: 2.2735 | Val Loss: 2.3160 | mIoU: 0.0015 | Time: 85.2s
09:30:43 [INFO]   Epoch 10 [20/63] Loss: 2.2466
09:31:05 [INFO]   Epoch 10 [40/63] Loss: 2.2436
09:31:25 [INFO]   Epoch 10 [60/63] Loss: 2.2378
09:31:49 [INFO] Epoch  10 | Train Loss: 2.2385 | Val Loss: 2.2235 | mIoU: 0.0012 | Time: 86.1s
09:32:09 [INFO]   Epoch 11 [20/63] Loss: 2.2130
09:32:28 [INFO]   Epoch 11 [40/63] Loss: 2.2071
09:32:49 [INFO]   Epoch 11 [60/63] Loss: 2.2092
09:33:11 [INFO] Epoch  11 | Train Loss: 2.2096 | Val Loss: 2.2015 | mIoU: 0.0000 | Time: 82.4s
09:33:34 [INFO]   Epoch 12 [20/63] Loss: 2.2049
09:33:53 [INFO]   Epoch 12 [40/63] Loss: 2.1936
09:34:13 [INFO]   Epoch 12 [60/63] Loss: 2.1889
09:34:36 [INFO] Epoch  12 | Train Loss: 2.1887 | Val Loss: 2.1920 | mIoU: 0.0007 | Time: 84.9s
09:34:56 [INFO]   Epoch 13 [20/63] Loss: 2.1799
09:35:16 [INFO]   Epoch 13 [40/63] Loss: 2.1740
09:35:36 [INFO]   Epoch 13 [60/63] Loss: 2.1763
09:35:59 [INFO] Epoch  13 | Train Loss: 2.1749 | Val Loss: 2.1765 | mIoU: 0.0000 | Time: 82.9s
09:36:21 [INFO]   Epoch 14 [20/63] Loss: 2.1758
09:36:41 [INFO]   Epoch 14 [40/63] Loss: 2.1631
09:37:01 [INFO]   Epoch 14 [60/63] Loss: 2.1589
09:37:24 [INFO] Epoch  14 | Train Loss: 2.1620 | Val Loss: 2.2247 | mIoU: 0.0078 | Time: 84.7s
09:37:46 [INFO]   Epoch 15 [20/63] Loss: 2.1430
09:38:06 [INFO]   Epoch 15 [40/63] Loss: 2.1475
09:38:25 [INFO]   Epoch 15 [60/63] Loss: 2.1561
09:38:49 [INFO] Epoch  15 | Train Loss: 2.1541 | Val Loss: 2.1815 | mIoU: 0.0001 | Time: 85.6s
09:39:10 [INFO]   Epoch 16 [20/63] Loss: 2.1616
09:39:29 [INFO]   Epoch 16 [40/63] Loss: 2.1638
09:39:51 [INFO]   Epoch 16 [60/63] Loss: 2.1524
09:40:13 [INFO] Epoch  16 | Train Loss: 2.1513 | Val Loss: 2.1792 | mIoU: 0.0000 | Time: 83.8s
09:40:35 [INFO]   Epoch 17 [20/63] Loss: 2.1447
09:40:54 [INFO]   Epoch 17 [40/63] Loss: 2.1439
09:41:15 [INFO]   Epoch 17 [60/63] Loss: 2.1476
09:41:39 [INFO] Epoch  17 | Train Loss: 2.1463 | Val Loss: 2.2154 | mIoU: 0.0026 | Time: 85.4s
09:42:00 [INFO]   Epoch 18 [20/63] Loss: 2.1311
09:42:21 [INFO]   Epoch 18 [40/63] Loss: 2.1396
09:42:40 [INFO]   Epoch 18 [60/63] Loss: 2.1382
09:43:04 [INFO] Epoch  18 | Train Loss: 2.1379 | Val Loss: 2.4271 | mIoU: 0.0086 | Time: 85.0s
09:43:11 [INFO]   ★ New best model saved: mIoU=0.0086 → ./checkpoints/fusion_net_best.pth
09:43:34 [INFO]   Epoch 19 [20/63] Loss: 2.1294
09:43:54 [INFO]   Epoch 19 [40/63] Loss: 2.1233
09:44:13 [INFO]   Epoch 19 [60/63] Loss: 2.1263
09:44:37 [INFO] Epoch  19 | Train Loss: 2.1250 | Val Loss: 2.1506 | mIoU: 0.0146 | Time: 85.8s
09:44:39 [INFO]   ★ New best model saved: mIoU=0.0146 → ./checkpoints/fusion_net_best.pth
09:45:00 [INFO]   Epoch 20 [20/63] Loss: 2.1209
09:45:22 [INFO]   Epoch 20 [40/63] Loss: 2.1171
09:45:42 [INFO]   Epoch 20 [60/63] Loss: 2.1244
09:46:05 [INFO] Epoch  20 | Train Loss: 2.1251 | Val Loss: 2.4300 | mIoU: 0.0093 | Time: 86.3s
09:46:25 [INFO]   Epoch 21 [20/63] Loss: 2.1139
09:46:44 [INFO]   Epoch 21 [40/63] Loss: 2.1243
09:47:06 [INFO]   Epoch 21 [60/63] Loss: 2.1175
09:47:28 [INFO] Epoch  21 | Train Loss: 2.1168 | Val Loss: 2.1769 | mIoU: 0.0000 | Time: 83.2s
09:47:50 [INFO]   Epoch 22 [20/63] Loss: 2.1217
09:48:10 [INFO]   Epoch 22 [40/63] Loss: 2.1253
09:48:29 [INFO]   Epoch 22 [60/63] Loss: 2.1226
09:48:53 [INFO] Epoch  22 | Train Loss: 2.1229 | Val Loss: 2.2512 | mIoU: 0.0170 | Time: 84.6s
09:48:56 [INFO]   ★ New best model saved: mIoU=0.0170 → ./checkpoints/fusion_net_best.pth
09:49:19 [INFO]   Epoch 23 [20/63] Loss: 2.1069
09:49:38 [INFO]   Epoch 23 [40/63] Loss: 2.1134
09:49:58 [INFO]   Epoch 23 [60/63] Loss: 2.1137
09:50:22 [INFO] Epoch  23 | Train Loss: 2.1135 | Val Loss: 2.1920 | mIoU: 0.0001 | Time: 85.4s
09:50:42 [INFO]   Epoch 24 [20/63] Loss: 2.1028
09:51:01 [INFO]   Epoch 24 [40/63] Loss: 2.1155
09:51:21 [INFO]   Epoch 24 [60/63] Loss: 2.1179
09:51:44 [INFO] Epoch  24 | Train Loss: 2.1175 | Val Loss: 2.2113 | mIoU: 0.0264 | Time: 82.2s
09:51:46 [INFO]   ★ New best model saved: mIoU=0.0264 → ./checkpoints/fusion_net_best.pth
09:52:09 [INFO]   Epoch 25 [20/63] Loss: 2.1117
09:52:28 [INFO]   Epoch 25 [40/63] Loss: 2.1123
09:52:48 [INFO]   Epoch 25 [60/63] Loss: 2.1126
09:53:11 [INFO] Epoch  25 | Train Loss: 2.1134 | Val Loss: 2.1937 | mIoU: 0.0016 | Time: 85.5s
09:53:32 [INFO]   Epoch 26 [20/63] Loss: 2.1129
09:53:53 [INFO]   Epoch 26 [40/63] Loss: 2.1143
09:54:12 [INFO]   Epoch 26 [60/63] Loss: 2.1083
09:54:36 [INFO] Epoch  26 | Train Loss: 2.1063 | Val Loss: 2.1633 | mIoU: 0.0215 | Time: 84.9s
09:54:57 [INFO]   Epoch 27 [20/63] Loss: 2.1041
09:55:16 [INFO]   Epoch 27 [40/63] Loss: 2.1103
09:55:37 [INFO]   Epoch 27 [60/63] Loss: 2.1043
09:55:59 [INFO] Epoch  27 | Train Loss: 2.1054 | Val Loss: 2.9507 | mIoU: 0.0083 | Time: 82.5s
09:56:21 [INFO]   Epoch 28 [20/63] Loss: 2.1022
09:56:40 [INFO]   Epoch 28 [40/63] Loss: 2.1026
09:56:59 [INFO]   Epoch 28 [60/63] Loss: 2.0992
09:57:23 [INFO] Epoch  28 | Train Loss: 2.1003 | Val Loss: 2.1581 | mIoU: 0.0333 | Time: 84.0s
09:57:26 [INFO]   ★ New best model saved: mIoU=0.0333 → ./checkpoints/fusion_net_best.pth
09:57:48 [INFO]   Epoch 29 [20/63] Loss: 2.0812
09:58:07 [INFO]   Epoch 29 [40/63] Loss: 2.0937
09:58:26 [INFO]   Epoch 29 [60/63] Loss: 2.0969
09:58:49 [INFO] Epoch  29 | Train Loss: 2.0972 | Val Loss: 2.1741 | mIoU: 0.0183 | Time: 82.6s
09:59:09 [INFO]   Epoch 30 [20/63] Loss: 2.0862
09:59:27 [INFO]   Epoch 30 [40/63] Loss: 2.1068
09:59:48 [INFO]   Epoch 30 [60/63] Loss: 2.0962
10:00:09 [INFO] Epoch  30 | Train Loss: 2.0970 | Val Loss: 2.1206 | mIoU: 0.0490 | Time: 80.5s
10:00:11 [INFO]   ★ New best model saved: mIoU=0.0490 → ./checkpoints/fusion_net_best.pth
10:00:34 [INFO]   Epoch 31 [20/63] Loss: 2.1110
10:00:53 [INFO]   Epoch 31 [40/63] Loss: 2.1028
10:01:12 [INFO]   Epoch 31 [60/63] Loss: 2.1030
10:01:34 [INFO] Epoch  31 | Train Loss: 2.1015 | Val Loss: 2.1212 | mIoU: 0.0510 | Time: 83.1s
10:01:38 [INFO]   ★ New best model saved: mIoU=0.0510 → ./checkpoints/fusion_net_best.pth
10:02:00 [INFO]   Epoch 32 [20/63] Loss: 2.0791
10:02:19 [INFO]   Epoch 32 [40/63] Loss: 2.0941
10:02:37 [INFO]   Epoch 32 [60/63] Loss: 2.0991
10:03:00 [INFO] Epoch  32 | Train Loss: 2.0982 | Val Loss: 2.1190 | mIoU: 0.0495 | Time: 82.1s
10:03:20 [INFO]   Epoch 33 [20/63] Loss: 2.0875
10:03:38 [INFO]   Epoch 33 [40/63] Loss: 2.0875
10:03:58 [INFO]   Epoch 33 [60/63] Loss: 2.1000
10:04:20 [INFO] Epoch  33 | Train Loss: 2.1001 | Val Loss: 2.1535 | mIoU: 0.0317 | Time: 79.8s
10:04:40 [INFO]   Epoch 34 [20/63] Loss: 2.0758
10:05:00 [INFO]   Epoch 34 [40/63] Loss: 2.0897
10:05:19 [INFO]   Epoch 34 [60/63] Loss: 2.0887
10:05:42 [INFO] Epoch  34 | Train Loss: 2.0882 | Val Loss: 2.1161 | mIoU: 0.0530 | Time: 81.8s
10:05:44 [INFO]   ★ New best model saved: mIoU=0.0530 → ./checkpoints/fusion_net_best.pth
10:06:05 [INFO]   Epoch 35 [20/63] Loss: 2.0940
10:06:24 [INFO]   Epoch 35 [40/63] Loss: 2.0892
10:06:44 [INFO]   Epoch 35 [60/63] Loss: 2.0890
10:07:06 [INFO] Epoch  35 | Train Loss: 2.0884 | Val Loss: 2.1224 | mIoU: 0.0480 | Time: 82.4s
10:07:28 [INFO]   Epoch 36 [20/63] Loss: 2.0793
10:07:47 [INFO]   Epoch 36 [40/63] Loss: 2.0942
10:08:06 [INFO]   Epoch 36 [60/63] Loss: 2.0985
10:08:29 [INFO] Epoch  36 | Train Loss: 2.0974 | Val Loss: 2.1157 | mIoU: 0.0543 | Time: 82.7s
10:08:34 [INFO]   ★ New best model saved: mIoU=0.0543 → ./checkpoints/fusion_net_best.pth
10:08:55 [INFO]   Epoch 37 [20/63] Loss: 2.0846
10:09:15 [INFO]   Epoch 37 [40/63] Loss: 2.0923
10:09:35 [INFO]   Epoch 37 [60/63] Loss: 2.0925
10:09:58 [INFO] Epoch  37 | Train Loss: 2.0948 | Val Loss: 2.1138 | mIoU: 0.0539 | Time: 84.0s
10:10:18 [INFO]   Epoch 38 [20/63] Loss: 2.0791
10:10:36 [INFO]   Epoch 38 [40/63] Loss: 2.0975
10:10:57 [INFO]   Epoch 38 [60/63] Loss: 2.0964
10:11:19 [INFO] Epoch  38 | Train Loss: 2.0953 | Val Loss: 2.1138 | mIoU: 0.0531 | Time: 81.1s
10:11:40 [INFO]   Epoch 39 [20/63] Loss: 2.0799
10:12:00 [INFO]   Epoch 39 [40/63] Loss: 2.0868
10:12:19 [INFO]   Epoch 39 [60/63] Loss: 2.0899
10:12:42 [INFO] Epoch  39 | Train Loss: 2.0931 | Val Loss: 2.1156 | mIoU: 0.0531 | Time: 83.0s
10:12:42 [INFO] ============================================================
10:12:42 [INFO] Training complete. Best mIoU: 0.0543
10:12:42 [INFO] Best checkpoint: ./checkpoints/fusion_net_best.pth
10:12:42 [INFO] ============================================================
🔬 Training Baseline U-Net (25 epochs)...
10:12:50 [INFO] Device: cuda
10:12:50 [INFO] Creating DataLoaders...
10:12:50 [INFO] Built manifest: 801 samples from 2 dirs
10:12:50 [INFO]   Mask lookup: 801 found, 0 missing (100% coverage)
10:12:50 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
10:12:52 [INFO] Created DataLoaders: train=510, val=202, test=89
10:12:52 [INFO] Running data sanity check...
10:12:59 [INFO]   Sanity check: 15/25 class-batch slots have positive pixels (checked 40 samples)
10:12:59 [INFO] Building model: unet
10:13:01 [INFO] Parameters: 47.69M (47689733 trainable)
10:13:01 [INFO] ============================================================
10:13:01 [INFO] Starting training: unet for 25 epochs
10:13:01 [INFO] ============================================================
10:13:21 [INFO]   Epoch 0 [20/63] Loss: 2.5983
10:13:40 [INFO]   Epoch 0 [40/63] Loss: 2.5926
10:13:58 [INFO]   Epoch 0 [60/63] Loss: 2.5885
10:14:19 [INFO] Epoch   0 | Train Loss: 2.5877 | Val Loss: 2.5590 | mIoU: 0.0083 | Time: 77.1s
10:14:22 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/unet_net_best.pth
10:14:42 [INFO]   Epoch 1 [20/63] Loss: 2.5685
10:15:00 [INFO]   Epoch 1 [40/63] Loss: 2.5608
10:15:17 [INFO]   Epoch 1 [60/63] Loss: 2.5545
10:15:39 [INFO] Epoch   1 | Train Loss: 2.5530 | Val Loss: 2.5190 | mIoU: 0.0083 | Time: 76.8s
10:15:46 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/unet_net_best.pth
10:16:06 [INFO]   Epoch 2 [20/63] Loss: 2.5253
10:16:24 [INFO]   Epoch 2 [40/63] Loss: 2.5110
10:16:42 [INFO]   Epoch 2 [60/63] Loss: 2.5001
10:17:04 [INFO] Epoch   2 | Train Loss: 2.4987 | Val Loss: 2.4781 | mIoU: 0.0083 | Time: 77.9s
10:17:06 [INFO]   ★ New best model saved: mIoU=0.0083 → ./checkpoints/unet_net_best.pth
10:17:25 [INFO]   Epoch 3 [20/63] Loss: 2.4628
10:17:43 [INFO]   Epoch 3 [40/63] Loss: 2.4557
10:18:00 [INFO]   Epoch 3 [60/63] Loss: 2.4517
10:18:22 [INFO] Epoch   3 | Train Loss: 2.4504 | Val Loss: 2.4353 | mIoU: 0.0086 | Time: 76.4s
10:18:26 [INFO]   ★ New best model saved: mIoU=0.0086 → ./checkpoints/unet_net_best.pth
10:18:45 [INFO]   Epoch 4 [20/63] Loss: 2.4228
10:19:02 [INFO]   Epoch 4 [40/63] Loss: 2.4139
10:19:21 [INFO]   Epoch 4 [60/63] Loss: 2.4084
10:19:42 [INFO] Epoch   4 | Train Loss: 2.4076 | Val Loss: 2.3767 | mIoU: 0.0079 | Time: 75.8s
10:20:00 [INFO]   Epoch 5 [20/63] Loss: 2.3739
10:20:19 [INFO]   Epoch 5 [40/63] Loss: 2.3632
10:20:37 [INFO]   Epoch 5 [60/63] Loss: 2.3580
10:20:57 [INFO] Epoch   5 | Train Loss: 2.3570 | Val Loss: 2.2921 | mIoU: 0.0057 | Time: 75.8s
10:21:16 [INFO]   Epoch 6 [20/63] Loss: 2.3246
10:21:36 [INFO]   Epoch 6 [40/63] Loss: 2.3161
10:21:54 [INFO]   Epoch 6 [60/63] Loss: 2.3101
10:22:13 [INFO] Epoch   6 | Train Loss: 2.3081 | Val Loss: 2.2787 | mIoU: 0.0018 | Time: 76.1s
10:22:34 [INFO]   Epoch 7 [20/63] Loss: 2.2856
10:22:52 [INFO]   Epoch 7 [40/63] Loss: 2.2734
10:23:09 [INFO]   Epoch 7 [60/63] Loss: 2.2694
10:23:30 [INFO] Epoch   7 | Train Loss: 2.2678 | Val Loss: 2.2761 | mIoU: 0.0008 | Time: 76.3s
10:23:49 [INFO]   Epoch 8 [20/63] Loss: 2.2475
10:24:06 [INFO]   Epoch 8 [40/63] Loss: 2.2369
10:24:24 [INFO]   Epoch 8 [60/63] Loss: 2.2366
10:24:46 [INFO] Epoch   8 | Train Loss: 2.2349 | Val Loss: 2.2233 | mIoU: 0.0000 | Time: 75.8s
10:25:04 [INFO]   Epoch 9 [20/63] Loss: 2.2152
10:25:22 [INFO]   Epoch 9 [40/63] Loss: 2.2125
10:25:39 [INFO]   Epoch 9 [60/63] Loss: 2.2097
10:26:01 [INFO] Epoch   9 | Train Loss: 2.2113 | Val Loss: 2.2172 | mIoU: 0.0000 | Time: 75.5s
10:26:20 [INFO]   Epoch 10 [20/63] Loss: 2.2080
10:26:38 [INFO]   Epoch 10 [40/63] Loss: 2.1971
10:26:55 [INFO]   Epoch 10 [60/63] Loss: 2.1939
10:27:17 [INFO] Epoch  10 | Train Loss: 2.1939 | Val Loss: 2.2123 | mIoU: 0.0000 | Time: 76.0s
10:27:35 [INFO]   Epoch 11 [20/63] Loss: 2.1726
10:27:53 [INFO]   Epoch 11 [40/63] Loss: 2.1733
10:28:12 [INFO]   Epoch 11 [60/63] Loss: 2.1790
10:28:32 [INFO] Epoch  11 | Train Loss: 2.1769 | Val Loss: 2.2393 | mIoU: 0.0010 | Time: 74.8s
10:28:50 [INFO]   Epoch 12 [20/63] Loss: 2.1569
10:29:08 [INFO]   Epoch 12 [40/63] Loss: 2.1628
10:29:27 [INFO]   Epoch 12 [60/63] Loss: 2.1640
10:29:47 [INFO] Epoch  12 | Train Loss: 2.1660 | Val Loss: 2.1796 | mIoU: 0.0000 | Time: 75.4s
10:30:05 [INFO]   Epoch 13 [20/63] Loss: 2.1575
10:30:23 [INFO]   Epoch 13 [40/63] Loss: 2.1544
10:30:41 [INFO]   Epoch 13 [60/63] Loss: 2.1552
10:31:02 [INFO] Epoch  13 | Train Loss: 2.1591 | Val Loss: 2.2020 | mIoU: 0.0021 | Time: 74.5s
10:31:02 [INFO] Early stopping at epoch 13 (patience=10)
10:31:02 [INFO] ============================================================
10:31:02 [INFO] Training complete. Best mIoU: 0.0086
10:31:02 [INFO] Best checkpoint: ./checkpoints/unet_net_best.pth
10:31:02 [INFO] ============================================================
📊 Evaluating models with TTA & SpaceNet 8 Overlap Heuristics...
10:31:09 [INFO] Device: cuda | TTA: True | Overlap Heuristics: True
10:31:09 [INFO] Built manifest: 801 samples from 2 dirs
10:31:09 [INFO]   Mask lookup: 801 found, 0 missing (100% coverage)
10:31:09 [INFO] Dataset split: train=510, val=202, test=89
/usr/local/lib/python3.13/dist-packages/albumentations/core/validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
10:31:11 [INFO] Created DataLoaders: train=510, val=202, test=89
10:31:11 [INFO] Evaluating on val set: 202 samples
10:31:11 [INFO] 
--- Evaluating U-Net Baseline ---
10:31:14 [INFO] Loaded unet from ./checkpoints/unet_net_best.pth (epoch 3, best_iou=0.008565227548633566)
10:31:19 [INFO]   [U-Net] Evaluated 10/51 batches
10:31:22 [INFO]   [U-Net] Evaluated 20/51 batches
10:31:27 [INFO]   [U-Net] Evaluated 30/51 batches
10:31:30 [INFO]   [U-Net] Evaluated 40/51 batches
10:31:34 [INFO]   [U-Net] Evaluated 50/51 batches
10:31:34 [INFO] 
--- Evaluating Siamese Fusion ---
10:31:38 [INFO] Loaded fusion from ./checkpoints/fusion_net_best.pth (epoch 36, best_iou=0.05426615981336956)
10:31:46 [INFO]   [Fusion] Evaluated 10/51 batches
10:31:53 [INFO]   [Fusion] Evaluated 20/51 batches
10:32:00 [INFO]   [Fusion] Evaluated 30/51 batches
10:32:07 [INFO]   [Fusion] Evaluated 40/51 batches
10:32:14 [INFO]   [Fusion] Evaluated 50/51 batches

==========================================================================================
MODEL COMPARISON — SpaceNet 8 Flood Mapping
==========================================================================================
Metric                   |        UNet Baseline |       Siamese Fusion | {'Δ (Fusion - UNet)':>20}
------------------------------------------------------------------------------------------
iou_building             |              0.0153 |              0.0951 | +            0.0797 
f1_building              |              0.0296 |              0.1589 | +            0.1294 
------------------------------------------------------------------------------------------
iou_road                 |              0.0124 |              0.0714 | +            0.0589 
f1_road                  |              0.0245 |              0.1264 | +            0.1020 
------------------------------------------------------------------------------------------
iou_flood                |              0.0069 |              0.0329 | +            0.0260 
f1_flood                 |              0.0135 |              0.0560 | +            0.0425 
------------------------------------------------------------------------------------------
iou_flooded_building     |              0.0042 |              0.0254 | +            0.0212 
f1_flooded_building      |              0.0082 |              0.0447 | +            0.0365 
------------------------------------------------------------------------------------------
iou_flooded_road         |              0.0023 |              0.0134 | +            0.0111 
f1_flooded_road          |              0.0046 |              0.0217 | +            0.0172 
------------------------------------------------------------------------------------------
★ mean_iou               |              0.0082 |              0.0476 | +            0.0394 
★ mean_f1                |              0.0160 |              0.0815 | +            0.0655 
==========================================================================================
10:32:14 [INFO] Results saved to ./logs/final_eval_benchmark.json
💾 Backing up trained checkpoints & benchmarks to Google Drive...
✅ Successfully backed up to /content/drive/MyDrive/SpaceNet8_Checkpoints/
==========================================================
🎉 Colab Training & Benchmark Complete!
Checkpoints: ./checkpoints/
Logs:        ./logs/
==========================================================