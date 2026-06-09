import cv2
import numpy as np
import time

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def test_homography_loop():
    cap = cv2.VideoCapture(0)
    
    crop_size = 128
    dst_matrix = np.array([[0,0], [crop_size-1,0], [crop_size-1,crop_size-1], [0,crop_size-1]], dtype="float32")
    
    print("Starting Mock Test. We are simulating a skewed target...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        #frame info described here:
        h, w = frame.shape[:2] 
        mock_obb_corners = np.array([
            [w//3, h//3],       # Top-Left (Skewed)
            [2*w//3 - 30, h//3 + 20],  # Top-Right (Skewed)
            [2*w//3, 2*h//3],   # Bottom-Right (Skewed)
            [w//3 + 40, 2*h//3 - 10]   # Bottom-Left (Skewed)
        ], dtype="float32")
      
        rect = order_points(mock_obb_corners)
        
        h_matrix = cv2.getPerspectiveTransform(rect, dst_matrix)
        flat_crop = cv2.warpPerspective(frame, h_matrix, (crop_size, crop_size))
        
        draw_pts = rect.astype(int)
        cv2.polylines(frame, [draw_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        for idx, pt in enumerate(draw_pts):
            cv2.putText(frame, f"Corner {idx}", tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        cv2.imshow("Main Live Feed (OBB Mocked)", frame)
        cv2.imshow("Flattened Crop (Ready for Classifier)", flat_crop)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_homography_loop()
