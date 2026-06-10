import cv2
import numpy as np
from ultralytics import YOLO
import json
from async_camera import AsyncCamera

# Set to False on competition day to disable display window and save CPU cycles
DEBUG = True 

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def run_vision_node():
    # engine is compiled directly on Jetson
    model = YOLO('best.engine', task='obb') 
    
    # Initialize CSI GStreamer for our webcam
    cam = AsyncCamera(camera_type="csi", sensor_id=0, width=1280, height=720, fps=60)
    
    crop_size = 128
    dst_matrix = np.array([[0,0], [crop_size,0], [crop_size,crop_size], [0,crop_size]], dtype="float32")

    print("Master Vision Node Active. Target Prioritization Enabled.")
    
    while True:
        ret, frame = cam.read()
        if not ret: continue
        
        # ByteTrack tracks object IDs across frames. Was used since this is better than BoT-SORT
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)[0]
        
        targets_list = []

        if results.obb is not None:
            corners_array = results.obb.xyxyxyxy.cpu().numpy()
            classes = results.obb.cls.cpu().numpy()
            ids = results.obb.id.cpu().numpy() if results.obb.id is not None else [-1] * len(classes)

            for corners, cls_id, obj_id in zip(corners_array, classes, ids):
                pts = np.array(corners, dtype="float32")
                rect = order_points(pts)
                
                cx = int(np.mean(rect[:, 0]))
                cy = int(np.mean(rect[:, 1]))
                
                # (smaller area = farther away) (based on OBB)
                area = cv2.contourArea(rect.astype(int))
                distance_score = 1.0 / area if area > 0 else float('inf')

                class_name = results.names[int(cls_id)]
                
                targets_list.append({
                    "id": int(obj_id),
                    "class": class_name,
                    "cx": cx,
                    "cy": cy,
                    "distance_metric": distance_score,
                    "rect": rect
                })

        if targets_list:
            # Sort targets so the closest target (largest area) is first
            targets_list.sort(key=lambda t: t["distance_metric"])
            best_target = targets_list[0] 
            
           
            serialized_packet = json.dumps({
                "id": best_target["id"],
                "class": best_target["class"],
                "cx": best_target["cx"],
                "cy": best_target["cy"]
            })
            
            # Write to serial UART
            print(f"UART SEND -> {serialized_packet}")

            # foe dat debug
            if DEBUG:
                draw_pts = best_target["rect"].astype(int)
                cv2.polylines(frame, [draw_pts], isClosed=True, color=(0, 255, 0), thickness=3)
                cv2.circle(frame, (best_target["cx"], best_target["cy"]), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"TARGET: {best_target['class']}", tuple(draw_pts[0]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if DEBUG:
            cv2.imshow("Jetson Ensemble Node", frame)
            if cv2.waitKey(1) == ord('q'): break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_node()
