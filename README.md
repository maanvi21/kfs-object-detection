# System Architecture Summary: Ensemble OBB & TensorRT Vision Pipeline

## 1. Data Curation & Transfer Learning
* **Class Consolidation:** The dataset was restructured from 9 classes to 6 consolidated classes (`Fist`, `Hand`, `R1`, `R2 Fake`, `R2 Real`, `Spear`), eliminating redundant color-split classes to resolve feature-space confusion during backpropagation.
* **Asymmetric Augmentation Filtering:** Standard dataset augmentations were modified to explicitly disable horizontal/vertical flips and 90-degree rotations, preventing label corruption and ensuring the neural network preserves the true spatial and semantic orientation of asymmetric characters on the targets.
* **Transfer Learning:** Training utilized a pre-trained `yolov8n-obb.pt` backbone as a warm-start checkpoint, allowing the model to leverage existing low-level spatial filters (edges, textures, gradients) while optimizing the output regression head for oriented target objects.

## 2. Oriented Bounding Boxes (OBB) & Geometry
* **Coordinate Regression:** The OBB model outputs five-parameter oriented bounding box vectors $(X_c, Y_c, W, H, \theta)$, enabling the system to localize targets with exact rotation angles.
* **Isolation of Background Noise:** By regressing rotated bounding boxes rather than standard horizontal bounding boxes, the system crops targets tightly, minimizing the inclusion of background noise and surrounding arena clutter within the cropped regions of interest.

## 3. Spatial Math & 3D Estimation (Homography & PnP)
* **Perspective Transform (Homography):** The system extracts the 4 rotated corners of the OBB, calculates a $3 \times 3$ homographic transformation matrix using OpenCV's `getPerspectiveTransform`, and applies `warpPerspective` to project skewed, angled 3D targets into flat, standardized $128 \times 128$ pixel 2D planes for classification.
* **Perspective-n-Point (PnP) Depth Solver:** By mapping the 2D pixel coordinates of the 4 OBB corners to the known 3D physical dimensions of the KFS target boxes ($350\text{mm} \times 350\text{mm} \times 350\text{mm}$) using the camera's intrinsic calibration matrix, the system runs `cv::solvePnP` to compute the real-world distance ($Z$) and rotation vector of the target relative to the camera lens.

## 4. Hardware-Accelerated I/O & Threading
* **Asynchronous Multi-Threading:** The Python implementation utilizes a LIFO queue thread (`queue.LifoQueue(maxsize=1)`) and the C++ engine utilizes a `std::mutex` with RAII `std::lock_guard` to separate the camera frame capture loop from the inference execution loop, preventing I/O blocking.
* **GStreamer Hardware Pipelines:** Camera frames are acquired using customized GStreamer strings (`nvarguscamerasrc` for CSI or `v4l2src` for USB) configured with NVIDIA's hardware-accelerated converter `nvvidconv` to copy frames directly to GPU memory (NVMM), eliminating CPU decoding overhead.

## 5. TensorRT & C++ Optimization
* **FP16 Quantization:** The model is compiled into an FP16 TensorRT engine (`.engine`) directly on the Jetson Nano, leveraging its 128 CUDA cores for half-precision floating-point execution.
* **Asynchronous CUDA Streams:** The C++ core utilizes `cudaStream_t` to overlap memory transfers with GPU compute, performing asynchronous host-to-device and device-to-host copies via `cudaMemcpyAsync` and executing the engine via `context->enqueueV2`.
* **Custom Rotated Non-Maximum Suppression (NMS):** Since TensorRT only executes raw neural network layers, a custom C++ post-processing function was implemented to sort candidates by confidence and perform Rotated NMS using OpenCV’s `rotatedRectangleIntersection` to eliminate overlapping duplicate boxes.
