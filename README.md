# FlashSAM
The FlashSAM is a CNN Segment Anything Model trained using only 2% of the SA-1B dataset published by SAM authors. FlashSAM achieves comparable performance with the SAM method at 50× higher run-time speed. Its backbone comes from [YOLO11](https://docs.ultralytics.com/zh/models/yolo11/).

# Install
We recommend [uv](https://docs.astral.sh/uv/) as the package manager, develop environment is Ubuntu 22.04 with cuda12.2
```
uv init -p 3.10
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# if you use conda, run pip install -r requirements.txt
```

# Download pretrained weight
FlashSAM: [link](https://drive.google.com/file/d/1c5_CWTob79kYfIY8GMcQeZdLfTG7dd1b/view?usp=drive_link)
Download and put it at FlashSAM/weights/

# Quick start
To infer in a script
```
uv run main.py
```

To infer in gradio for visualization
```
gradio app.py
```
