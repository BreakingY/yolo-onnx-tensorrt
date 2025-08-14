# predict.py

from ultralytics import YOLO

def predict():
    model = YOLO('runs/custom-yolov11s/weights/best.pt')

    model.predict(
        source='./test.jpg',  # 替换为你的测试图
        save=True,
        save_txt=True,
        save_conf=True,
        project='runs/predict',
        name='results',
        exist_ok=True
    )

    print("✅ 推理完成，结果保存在：runs/predict/results")

if __name__ == '__main__':
    predict()
