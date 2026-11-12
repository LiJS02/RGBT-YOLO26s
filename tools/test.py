from ultralytics import YOLO

model = YOLO("yolo26s.pt")

results = model("/home/ljs/datasets/M3FD_YOLO/ir/images/test/00221.png")

results[0].save(filename="result.jpg")