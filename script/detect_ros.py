#!/usr/bin/env python3

import roslib
roslib.load_manifest("yolov7_trt_ros")
import sys
import rospy
import cv2
import time
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
import numpy as np
# import ros_numpy as rnp
from cv_bridge import CvBridge
import threading

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt

from fastsam_ros_msgs.msg import Box, DetectionArray, Detection
from fastsam_ros_msgs.utils import *

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

        self.cfx = cuda.Device(0).make_context()
        # try:
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        # self.cfx = cuda.Device(0).make_context()
        # print(trt.__version__)

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        # except:
        #     self.cfx.pop()


    def infer(self, img):
        threading.Thread.__init__(self)
        self.cfx.push()

        self.inputs[0]['host'] = np.ravel(img)
        # self.cfx.push()
        # transfer data to the gpu
        # rospy.loginfo("a")
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # rospy.loginfo("b")
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # rospy.loginfo("c")
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # rospy.loginfo("d")
        # synchronize stream
        self.stream.synchronize()
        # rospy.loginfo("e")
        data = [out['host'] for out in self.outputs]
        self.cfx.pop()
        return data

    def destory(self):
        self.cfx.pop()

    def detect_video(self, video_path, conf=0.3, end2end=False):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.avi',fourcc,fps,(width,height))
        fps = 0
        import time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                                conf=conf, class_names=self.class_names)
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def inference(self, img_path, conf=0.3, end2end=False):
        origin_img = cv2.imread(img_path)
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        return origin_img

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(5):  # warmup
            _ = self.infer(img)

        print(5/(time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 4  # your model classes
        self.class_names = ["small_object", "big_object", "ship", "usv"]
        self.get_fps()
    
    def inference(self, img, conf=0.5, end2end=False):
        img, ratio = preproc(img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
            return (final_boxes, final_scores, final_cls_inds)
        return (None, None, None)
    
class Yolov7Detector:
    def __init__(self, model_path, threshold=0.7, compressed=False, display=False, image_in="image_in", image_out="image_out"):
        rospy.loginfo("Loading model {}".format(model_path))
        self.predictor = Predictor(model_path)
        self.conf = threshold
        self.compressed = compressed
        self.display = display
        self.img_bridge = CvBridge()
        self.n_classes = self.predictor.n_classes
        self.class_names = self.predictor.class_names

        if self.compressed:
            self.image_sub = rospy.Subscriber(image_in +'/compressed', CompressedImage, self.image_callback, queue_size=1)
        else:
            self.image_sub = rospy.Subscriber(image_in, Image, self.image_callback, queue_size=1)

        if self.display:
            if self.compressed:
                self.image_pub = rospy.Publisher(image_out + "/compressed", CompressedImage, queue_size=1)
            else:
                self.image_pub = rospy.Publisher(image_out, Image, queue_size=1)

        self.detection_pub = rospy.Publisher("~detections", Detection2DArray, queue_size=1)
        self.latency_pub = rospy.Publisher("~latency", Float64, queue_size=1)

    def image_callback(self, msg):
        self.frame_id = msg.header.frame_id

        
        if self.compressed:
            np_image = np.frombuffer(msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)
        else:
            cv_image = self.img_bridge.imgmsg_to_cv2(msg, "rgb8")
            if (cv_image.shape[2] == 4):
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)

        t0 = time.perf_counter()
        self.cv_image = cv_image

        boxes, scores, idxs = self.predictor.inference(self.cv_image, self.conf, end2end=True)
        
        detections = Detection2DArray()
        detections.header.stamp = msg.header.stamp
        detections.header.frame_id = msg.header.frame_id
        for i in range(len(boxes)):
            if (scores[i] <= self.conf):
                continue

            min_x, min_y, max_x, max_y = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
            center_x = (max_x+min_x)/2.0
            center_y = (max_y+min_y)/2.0
            height = max_y-min_y
            width = max_x-min_x

            bbox = BoundingBox2D()
            bbox.center.x = center_x
            bbox.center.y = center_y
            bbox.size_x = width
            bbox.size_y = height

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(idxs[i])
            hypothesis.score = scores[i]
            hypothesis.pose.pose.position.x = center_x
            hypothesis.pose.pose.position.y = center_y
            
            crop = self.cv_image[int(min_y):int(max_y), int(min_x):int(max_x)].astype('uint8')

            instance = Detection2D()
            instance.header.stamp = msg.header.stamp
            instance.header.frame_id = msg.header.frame_id
            instance.results.append(hypothesis)
            instance.bbox = bbox
            # instance.source_img.data = np.array(buffer).tostring()
            instance.source_img.header.frame_id = msg.header.frame_id
            instance.source_img.header.stamp = msg.header.stamp

            detections.detections.append(instance)
        self.detection_pub.publish(detections)

        latency = rospy.Time.now() - msg.header.stamp
        latency_msg = Float64()
        latency_msg.data = latency.to_sec()
        self.latency_pub.publish(latency_msg)            

        # det_msg = DetectionArray()
        # det_msg.header.stamp = msg.header.stamp
        # det_msg.header.frame_id = msg.header.frame_id
        # # det_msg.image = msg
        # det_msg.image_height = self.cv_image.shape[0]
        # det_msg.image_witdh = self.cv_image.shape[1]

        # for i in range(len(boxes)):
        #     min_x, min_y, max_x, max_y = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
        #     center_x = (max_x+min_x)/2.0
        #     center_y = (max_y+min_y)/2.0
        #     height = max_y-min_y
        #     width = max_x-min_x

        #     bbox = Box()
        #     bbox.c_x = center_x
        #     bbox.c_y = center_y
        #     bbox.width = width
        #     bbox.height = height

        #     det = Detection()
        #     det.class_id = int(idxs[i])
        #     det.class_name = self.predictor.class_names[det.class_id]
        #     det.score = scores[i]
        #     det.ori_box = bbox
        #     det_msg.detections.append(det)
        # self.detection_pub.publish(det_msg)

        if self.display:
            display_img = vis(self.cv_image, boxes, scores, idxs, conf=self.conf, class_names=self.predictor.class_names)
            if self.compressed:
                if(self.image_pub.get_num_connections() > 0):
                    pub_msg = CompressedImage()
                    pub_msg.header.stamp = msg.header.stamp
                    pub_msg.format = "jpeg"
                    pub_msg.data = np.array(cv2.imencode('.jpg', display_img)[1]).tostring()
                    # Publish new image
                    self.image_pub.publish(pub_msg)
            else:
                if(self.image_pub.get_num_connections() > 0):
                    pub_msg = self.img_bridge.cv2_to_imgmsg(self.cv_image, "rgb8")
                    pub_msg.header.frame_id = self.frame_id
                    pub_msg.header.stamp = msg.header.stamp
                    self.image_pub.publish(pub_msg)

<<<<<<< HEAD
        # print('YOLO :', (time.perf_counter() - t0)*1000, 'ms')
=======
>>>>>>> a2e4f23c01cf29f92af6909f53ca1cb9bd9f79d2
        rospy.loginfo_throttle(1.0, 'YOLO : {} ms'.format((time.perf_counter() - t0)*1000))

def main(args):
    rospy.init_node('detect', anonymous=True)
    
    model_path = rospy.get_param('~model_path')
    compressed = rospy.get_param('~compressed', default=False)
    threshold = rospy.get_param('~threshold', default=0.5)
    display = rospy.get_param('~display', default=False)
    image_in = rospy.get_param('~image_in', default="image_in")
    image_out = rospy.get_param('~image_out', default="image_out")

    detector = Yolov7Detector(model_path, threshold, compressed, display, image_in, image_out)
    rospy.loginfo("Running detector!")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        detector.predictor.destory()
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)