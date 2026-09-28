from ultralytics import YOLO


model = YOLO("yolo11n.pt")

results = model.predict(
    source="data/input/test.jpg",
    device=0,
    imgsz=640,
    conf=0.35,
    save=True,
    project="data/output",
    name="python_detection",
)

print("Algılama tamamlandı.")
print("Tespit edilen nesne sayısı:", len(results[0].boxes))