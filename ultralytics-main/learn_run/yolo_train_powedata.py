from ultralytics import YOLO
model = YOLO("yolov10n.pt") # load a pretrained model (recommended for training)
model.train(data="/home/e706/zhanxiangning/newtask/powerdatasets/powerdata.yaml", epochs=10, imgsz=640, name="yolov10n_powedata") # train the model