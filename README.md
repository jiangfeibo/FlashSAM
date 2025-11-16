# FlashSAM
The FlashSAM is a CNN Segment Anything Model trained using only 2% of the SA-1B dataset published by SAM authors. FlashSAM achieves comparable performance with the SAM method at 50× higher run-time speed. Its backbone comes from [YOLO11](https://docs.ultralytics.com/zh/models/yolo11/).

# Install
We recommend [uv](https://docs.astral.sh/uv/) as the package manager
```
uv init -p 3.10
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# if you use conda, run pip install -r requirements.txt
```

# Download pretrained weight
FlashSAM: [link]()

Clip: [link]()

Download and put them in FlashSAM/weights/

# Quick start
To infer in a script
```
uv run main.py
```

To infer in gradio for visualization
```
gradio app.py
```
