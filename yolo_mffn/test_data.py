from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolo11n.pt')  # or your custom model
    
    # Train the model
    results = model.train(
        data='/home/e706/zhanxiangning/newtask/powerdatasets/publicallpower.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda',  # or 'cpu'
        workers=8,
        project='dataset_test',
        name='yolo_exp1'
    )
#./e706/anaconda3/envs/dataset_test/lib/python3.14/site-packages/ultralytics/models/yolo/detect/
if __name__ == '__main__':
    main()