import sys
import cv2
import numpy as np
import time
import de
from infer_engine import InferEngine
from concurrent.futures import ThreadPoolExecutor

# 配置参数
IMAGE_PATHS = {
    'two': '/DEngine/data/111.png',
    'five': '/DEngine/data/222.png',
    'fist': '/DEngine/data/333.png'
}
MODEL_INPUT_SIZE = (416, 416)
DISPLAY_SIZE = (640, 480)
MAX_OVERLAY_SIZE = 200
CONF_THRESH = 0.75
IOU_THRESH = 0.4
TARGET_FPS = 10
FRAME_DELAY = 1.0 / TARGET_FPS
JPEG_QUALITY = 90

# 加载并预处理叠加图片
def load_overlay_images():
    overlay_imgs = {}
    for cls, path in IMAGE_PATHS.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            overlay_imgs[cls] = None
            continue
        
        # 确保4通道（BGRA）
        if img.ndim == 3 and img.shape[2] == 3:
            alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            img = np.dstack((img, alpha))
        elif img.ndim != 3 or img.shape[2] != 4:
            overlay_imgs[cls] = None
            continue
        
        # 缩放图片
        h, w = img.shape[:2]
        scale = min(MAX_OVERLAY_SIZE / w, MAX_OVERLAY_SIZE / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        overlay_imgs[cls] = img
    return overlay_imgs

overlay_imgs = load_overlay_images()

# 坐标转换: [x,y,w,h] -> [x1,y1,x2,y2]
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# 非极大值抑制
def py_cpu_nms(dets, thresh):
    if len(dets) == 0:
        return np.array([], dtype=np.int32)
    
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)

# 过滤检测框
def filter_box(org_box, conf_thres, iou_thres):
    org_box = np.squeeze(org_box)
    if org_box.size == 0:
        return np.array([])
    
    mask = org_box[..., 4] > conf_thres
    box = org_box[mask]
    if box.size == 0:
        return np.array([])
    
    cls_ids = np.argmax(box[..., 5:], axis=1).astype(int)
    unique_cls = np.unique(cls_ids)
    output = []
    
    for cls in unique_cls:
        cls_mask = cls_ids == cls
        cls_boxes = box[cls_mask][:, :6]
        cls_boxes[:, 5] = cls
        cls_boxes_xyxy = xywh2xyxy(cls_boxes)
        keep = py_cpu_nms(cls_boxes_xyxy, iou_thres)
        output.extend(cls_boxes_xyxy[keep])
    
    return np.array(output) if output else np.array([])

# 透明叠加函数
def overlay_transparent(background, overlay, center_x, center_y, scale=1.0):
    if overlay is None or background is None:
        return background
    
    h, w = overlay.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    a = overlay[:, :, 3]
    a = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)[1]
    
    bg_h, bg_w = background.shape[:2]
    x = max(0, min(center_x - new_w // 2, bg_w - new_w))
    y = max(0, min(center_y - new_h // 2, bg_h - new_h))
    
    roi = background[y:y+new_h, x:x+new_w]
    a_normalized = a / 255.0
    a_normalized = np.expand_dims(a_normalized, axis=2)
    
    overlay_rgb = overlay[:, :, :3]
    blended = (overlay_rgb * a_normalized + roi * (1 - a_normalized)).astype(np.uint8)
    background[y:y+new_h, x:x+new_w] = blended
    return background

# 异步推理函数
def async_infer(engine, img):
    format = de.PixelFormat.DE_PIX_FMT_RGB888_PLANE
    data = [(format, (1, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]), img)]
    output = engine.predict(data)[0]
    return filter_box(output, CONF_THRESH, IOU_THRESH)

# 主函数
def main():
    # 加载模型
    engine = InferEngine("/DEngine/net.bin", "/DEngine/model.bin", max_batch=1)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    # 摄像头参数设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 类别映射
    VALID_CLASSES = {'two': 1, 'five': 4, 'fist': 6}
    class_id_map = {v: k for k, v in VALID_CLASSES.items()}
    
    # 初始化线程池
    executor = ThreadPoolExecutor(max_workers=1)
    current_future = None
    last_results = np.array([])
    last_frame_time = 0
    
    while True:
        # 控制帧率
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)
        last_frame_time = time.time()
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 优化JPEG压缩质量
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        result, encimg = cv2.imencode('.jpg', frame, encode_param)
        if result:
            frame = cv2.imdecode(encimg, 1)
        
        # 预处理
        model_frame = cv2.resize(frame, MODEL_INPUT_SIZE)
        img = model_frame[:, :, ::-1].transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        # 异步推理
        if current_future is None:
            current_future = executor.submit(async_infer, engine, img)
        elif current_future.done():
            last_results = current_future.result()
            current_future = executor.submit(async_infer, engine, img)
        
        # 准备显示帧
        display_frame = cv2.resize(frame, DISPLAY_SIZE)
        scale_ratio = DISPLAY_SIZE[0] / MODEL_INPUT_SIZE[0]
        
        # 处理检测结果
        if len(last_results) > 0:
            for box in last_results:
                x1, y1, x2, y2 = map(int, box[:4])
                cls_id = int(box[5])
                
                if cls_id in class_id_map:
                    cls_name = class_id_map[cls_id]
                    center_x = int((x1 + x2) / 2 * scale_ratio)
                    center_y = int((y1 + y2) / 2 * scale_ratio)
                    box_size = min(x2 - x1, y2 - y1)
                    scale = (box_size / 100) * 1.2
                    scale = max(0.3, min(2.0, scale))
                    display_frame = overlay_transparent(
                        display_frame, overlay_imgs.get(cls_name), center_x, center_y, scale
                    )
        
        # 显示画面
        cv2.imshow("Gesture Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    engine.profile()
    executor.shutdown()

if __name__ == "__main__":
    main()