from ultralytics import YOLO

model = YOLO(model="yolo11x-seg.yaml", \
             )
model.train(data="sa.yaml", \
            epochs=100, \
            batch=32, \
            imgsz=1024, \
            overlap_mask=False, \
            save=True, \
            save_period=10, \
            device='0,1,2,3',\
            project='fastsam11_2%', \
            name='test', 
            val=False,)