import time
import sys
sys.path.insert(0, '..\Detectron2')
# Don't change the order of from detectron2.engine import DefaultPredictor or you will get an error!
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
class Detector:

    def __init__(self):
        """
        Initializes the Detector object by setting up the Detectron2 configuration,
        loading the pre-trained model, and preparing the KMeans object.
        """
        self.cfg = get_cfg()

        # load model config and pretrainec model
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x")

        # load model config and pretrainec model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

        # load KMeans and set the number of cluster
        self.clt = KMeans(n_clusters=2, n_init=10)


    def onImage(self, image):
        """
        Processes an image to detect objects, visualize the detections, and display the result.

        Parameters:
            image: The input image in BGR format.
        """
        predictions = self.predictor(image)
        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode.IMAGE_BW)

        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        output_image = output.get_image()[:, :, ::-1]

        # Get screen size
        screen_res = 1280, 720
        scale_width = screen_res[0] / output_image.shape[1]
        scale_height = screen_res[1] / output_image.shape[0]
        scale = min(scale_width, scale_height)

        # Calculate the new image size
        window_width = int(output_image.shape[1] * scale)
        window_height = int(output_image.shape[0] * scale)

        # Resize images to fit the screen
        resized_image = cv2.resize(output_image, (window_width, window_height))

        cv2.imwrite("../images/result.jpg",output_image)

        # Displaying the adjusted image
        cv2.imshow("result", resized_image)
        cv2.waitKey(0)


    def resultToList(self, image):
        """
        Processes the input image to detect objects and extracts information such as
        class, location, bounding box, confidence score, and color.

        Parameters:
            image: The input image in BGR format.

        Returns:
            A list of dictionaries, each containing information about detected objects.
        """
        list = []
        image_shape = image.shape[:2]
        predictions = self.predictor(image)

        instances = predictions['instances']

        # Access the pred_boxes property
        pred_boxes = instances.pred_boxes
        pred_classes = instances.pred_classes
        scores = instances.scores
        pred_masks = instances.pred_masks

        # Get the tensor data for the bounding box
        pred_boxes_tensor = pred_boxes.tensor

        # Convert tensor to numpy array for printing
        pred_boxes_array = pred_boxes_tensor.cpu().numpy()
        pred_classes_array = pred_classes.cpu().numpy()
        scores_array = scores.cpu().numpy()
        pred_masks_array = pred_masks.cpu().numpy()

        metadata = MetadataCatalog.get("coco_2017_train")
        # This will return a list with all the category names
        class_names = metadata.thing_classes

        # Histogram equalisation of images
        image_histogram = self.histogram_equalization_rgb(image)

        cv2.imwrite("./images/result.png",image_histogram)

        # Store visual information in a dictionary list
        for i in range(len(pred_classes_array)):
            mask = pred_masks_array[i]
            color = self.recoColour(mask, image_histogram)
            location = self.bbox_to_location(pred_boxes_array[i], image_shape)
            result = {
                'class':class_names[pred_classes_array[i]],
                'location': location ,
                'BBox': pred_boxes_array[i],
                'confidence': scores_array[i],
                'color': color,
            }
            list.append(result)
            # print(str(class_names[pred_classes_array[i]])+' '+str(pred_boxes_array[i])+' '+str(scores_array[i]))

        return list


    def bbox_to_location(self, box_array, image_shape):
        """
        Converts a bounding box array to a human-readable location.

        Parameters:
            box_array: Array containing the bounding box coordinates.
            image_shape: The shape of the input image.

        Returns:
            A string describing the location of the object within the image.
        """
        img_height, img_width = image_shape
        x_max = box_array[0]
        x_min = box_array[1]
        y_max = box_array[2]
        y_min = box_array[3]

        center_x, center_y = (x_max + x_min) / 2, (y_max + y_min) / 2

        # Judge of horizontal position
        if center_x < (img_width / 3):
            horizontal_position = "horizontal left"
        elif center_x > (2 * img_width / 3):
            horizontal_position = "horizontal right"
        else:
            horizontal_position = "horizontal middle"

        # Judge vertical position
        if center_y < (img_height / 3):
            vertical_position = "vertical top"
        elif center_y > (2 * img_height / 3):
            vertical_position = "vertical bottom"
        else:
            vertical_position = "vertical middle"

        return f"{horizontal_position} and {vertical_position}"


    def recoColour(self, mask, image):
        """
        Recognizes the dominant color of an object using KMeans clustering.

        Parameters:
            mask: The mask of the detected object.
            image: The input image in BGR format.

        Returns:
            The dominant color found in the specified object.
        """
        mask = mask.astype(np.uint8) * 255
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Apply mask
        target = cv2.bitwise_and(image, image, mask=mask)

        # Convert images to RGB type
        target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # Extracting objects at the pixel level using mask
        target = target_rgb[mask == 255]

        # Cluster the colours of objects to find the main two colours
        self.clt.fit(target)
        colors = self.convert_int(self.clt)


        # This example returns the indices of the closest colors. You might want to modify this to return the color names or any other format you need.
        return colors


    def histogram_equalization_rgb(self,image):
        """
        Applies histogram equalization to an RGB image to enhance contrast.

        Parameters:
            image: The input RGB image.

        Returns:
            The contrast-enhanced RGB image.
        """
        # Convert RGB images to YCrCb colour space (where Y is the luminance component)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        # Separating the channels of a YCrCb image
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)

        # Histogram equalisation of the luminance channel
        equalized_y_channel = cv2.equalizeHist(y_channel)

        # Recombine channels
        equalized_ycrcb_image = cv2.merge([equalized_y_channel, cr_channel, cb_channel])

        # Converting YCrCb images back to RGB colour space
        equalized_rgb_image = cv2.cvtColor(equalized_ycrcb_image, cv2.COLOR_YCrCb2RGB)

        return equalized_rgb_image


    def convert_int(self,k_cluster):
        """
        Convert the clustered centres to type int

        Parameters:
            k_cluster: The KMeans clustering object after fitting.

        Returns:
            An array of clustered centre

        """
        float_array = np.array(k_cluster.cluster_centers_)
        int_array = np.array(float_array).astype(int)
        return int_array

if __name__ == "__main__":
    start_time = time.time()
    detector = Detector()
    image = cv2.imread("./images/img.png")
    detector.resultToList(image)
    end_time = time.time()

    print(f"{end_time - start_time} s")
