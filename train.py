# train.py

from ultralytics import YOLO

def train():
    model = YOLO('yolo11s.pt')  # 使用官方预训练模型

    results = model.train(
        data='./data.yaml',
        epochs=50,
        imgsz=640,
        batch=18, # 必须是device的倍数
        project='runs',
        name='custom-yolov11s',
        exist_ok=True,
        device='0,1,2'  # 多卡训练
    )

    print(f"✅ 训练完成")

    # 训练完成后使用最优权重导出 ONNX
    best_model_path = 'runs/custom-yolov11s/weights/best.pt'
    best_model = YOLO(best_model_path)
    best_model.export(format='onnx', opset=12, dynamic=True, simplify=True, imgsz=640)


    print(f"✅ 模型已导出为 ONNX：{best_model_path.replace('.pt', '.onnx')}")

if __name__ == '__main__':
    train()
