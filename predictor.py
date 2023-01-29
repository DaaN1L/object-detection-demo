from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Colors, Annotator


colors = Colors()


class Predictor:
    def __init__(self, model):
        self.model = YOLO(model)
        self.names = self.model.names

    def predict(self, img, conf: float = 0.25, iou: float = 0.6):
        results = self.model.predict(img, conf=conf, iou=iou)
        return results

    def predict_and_annotate(self, img, conf: float = 0.25, iou: float = 0.6):
        result = self.predict(img, conf=conf, iou=iou)[0]
        annotator = Annotator(img)
        for boxes in result.boxes:
            box = boxes.xyxy.squeeze()
            cls, conf = boxes.cls.item(), boxes.conf.item()
            color = colors(cls)
            class_name = self.names[cls]
            label = f'{class_name} {conf:.2f}'
            annotator.box_label(box, label, color=color)

        return img
