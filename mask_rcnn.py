# https://pysource.com/instance-segmentation-mask-rcnn-with-python-and-opencv
import cv2
import numpy as np
import pyrealsense2 as rs
import collections

class MaskRCNN:
    def __init__(self):
        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("/home/ucsf/catkin_ws/src/kinova-ros/kinova_demo/nodes/kinova_demo/frozen_inference_graph_coco.pb",
                                            "/home/ucsf/catkin_ws/src/kinova-ros/kinova_demo/nodes/kinova_demo/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Generate random colors
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Conf threshold
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.classes = []
        with open("/home/ucsf/catkin_ws/src/kinova-ros/kinova_demo/nodes/kinova_demo/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        # Distances
        self.distances = []


    def detect_objects_mask(self, bgr_frame):
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])

        # Detect objects
        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]

        # Object Boxes
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            color = self.colors[int(class_id)]
            if score < self.detection_threshold:
                continue

            # Get box Coordinates
            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            self.obj_boxes.append([x, y, x2, y2])

            cx = (x + x2) // 2
            cy = (y + y2) // 2
            self.obj_centers.append((cx, cy))

            # append class
            self.obj_classes.append(class_id)

            # Contours
            # Get mask coordinates
            # Get the mask
            mask = masks[i, int(class_id)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.obj_contours.append(contours)

        return self.obj_boxes, self.obj_classes, self.obj_contours, self.obj_centers

    def draw_object_mask(self, bgr_frame):
        # loop through the detection
        for box, class_id, contours in zip(self.obj_boxes, self.obj_classes, self.obj_contours):
            x, y, x2, y2 = box
            roi = bgr_frame[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            color = self.colors[int(class_id)]

            roi_copy = np.zeros_like(roi)

            for cnt in contours:
                # cv2.f(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                cv2.drawContours(roi, [cnt], - 1, (int(color[0]), int(color[1]), int(color[2])), 3)
                cv2.fillPoly(roi_copy, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
                bgr_frame[y: y2, x: x2] = roi
        return bgr_frame

    def draw_object_info(self, bgr_frame, depth_frame, depth_intrin):
        # find the closest object
        rs_height = 13 # in cm
        close_x = 0
        close_y = 0
        close_z = 0
        close_class = ''
        # loop through the detection
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center

            depth_mm = depth_frame[cy, cx]
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [cy, cx], depth_mm)

            # for flat mounts
            real_x = round(depth_point[1],2)
            real_y = round(depth_point[0],2)
            real_dis = round(depth_point[2],2)
            real_xdis = np.sqrt(np.square(real_dis)-np.square(real_x))
            real_z = np.sqrt(np.square(real_xdis) - np.square(real_y))
            class_name = self.classes[int(class_id)]


            # real_x = round(depth_point[1],2)
            # real_y = round(depth_point[0],2)
            # real_z = round(depth_point[2],2)
            # real_xdist = np.sqrt(np.square(real_z) - np.square(real_x))
            # measure_dist = np.sqrt(np.square(real_z) - np.square(rs_height))


            class_name = self.classes[int(class_id)]


            if close_z == 0 or real_z < close_z:
                close_z = real_z
                close_y = real_y
                close_x = real_x
                close_class = class_name.capitalize()

            cv2.line(bgr_frame, (cx, y), (cx, y2), color, 1)
            cv2.line(bgr_frame, (x, cy), (x2, cy), color, 1)


            cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr_frame, "X: {} cm".format(real_x / 10), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
            cv2.putText(bgr_frame, "Y: {} cm".format(real_y / 10), (x + 5, y + 95), 0, 1.0, (255, 255, 255), 2)
            cv2.putText(bgr_frame, "Z: {} cm".format(round(real_z / 10,2)), (x + 5, y + 130), 0, 1.0, (255, 255, 255), 2)
            #cv2.putText(bgr_frame, "Distance: {} cm".format(round(measure_dist / 10,2)), (x + 5, y + 165), 0, 1.0, (255, 255, 255), 2)
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)





        return bgr_frame, close_x, close_y, close_z, close_class





