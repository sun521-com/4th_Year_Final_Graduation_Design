"""
This file is used to get all the class names of the COCO2017 dataset
"""
import sys
sys.path.insert(0, '..\Detectron2')
from detectron2.data import MetadataCatalog

metadata = MetadataCatalog.get("coco_2017_train")

class_names = metadata.thing_classes

print(class_names)