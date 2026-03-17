# FlashSAM training
1. 下载SA-1B的2%（https://ai.meta.com/datasets/segment-anything-downloads/）

2. 将SA-1B的label转为YOLO格式

```python
python rle2yolo.py
```

3. 训练

```python
python train_sa.py
```