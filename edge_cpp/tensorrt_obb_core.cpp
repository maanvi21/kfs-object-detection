#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

template <typename T>
struct TrtDeleter {
    void operator()(T* obj) const { if (obj) obj->destroy(); }
};

class TensorRT_OBB {
private:
    std::unique_ptr<IRuntime, TrtDeleter<IRuntime>> runtime;
    std::unique_ptr<ICudaEngine, TrtDeleter<ICudaEngine>> engine;
    std::unique_ptr<IExecutionContext, TrtDeleter<IExecutionContext>> context;
    
    cudaStream_t stream;
    void* buffers[2]; 
    
    const int inputSize = 1 * 3 * 640 * 640 * sizeof(float);
//based on YOLOv8 output structure
    const int outputSize = 1 * 20 * 8400 * sizeof(float); 
    const float confThreshold = 0.5f;
    const float nmsThreshold = 0.45f;
   //added Non-Maximum Suppression logic. What each part does has been described in the comments briefly.
    void performRotatedNMS(
        const std::vector<cv::RotatedRect>& boxes,
        const std::vector<float>& confidences,
        const std::vector<int>& classIds,
        std::vector<int>& keepIndices
    ) {
        std::vector<int> indices(boxes.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        // Sort indices by confidence descending
        std::sort(indices.begin(), indices.end(), [&confidences](int idx1, int idx2) {
            return confidences[idx1] > confidences[idx2];
        });

        std::vector<bool> isRemoved(boxes.size(), false);

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx1 = indices[i];
            if (isRemoved[idx1]) continue;

            keepIndices.push_back(idx1);

            for (size_t j = i + 1; j < indices.size(); ++j) {
                int idx2 = indices[j];
                if (isRemoved[idx2]) continue;

                // Only suppress boxes of the SAME predicted class
                if (classIds[idx1] != classIds[idx2]) continue;

                // Calculate Rotated Intersection Area
                std::vector<cv::Point2f> intersectionPoints;
                int intersectionType = cv::rotatedRectangleIntersection(boxes[idx1], boxes[idx2], intersectionPoints);

                if (intersectionType == cv::INTERSECT_NONE) continue;

                float intersectionArea = 0.0f;
                if (!intersectionPoints.empty()) {
                    intersectionArea = cv::contourArea(intersectionPoints);
                }

                // Calculate IoU (Intersection over Union)
                float area1 = boxes[idx1].size.width * boxes[idx1].size.height;
                float area2 = boxes[idx2].size.width * boxes[idx2].size.height;
                float unionArea = area1 + area2 - intersectionArea;

                float iou = (unionArea > 0) ? (intersectionArea / unionArea) : 0.0f;

                if (iou > nmsThreshold) {
                    isRemoved[idx2] = true;
                }
            }
        }
    }

public:
    TensorRT_OBB(const std::string& enginePath) {
        cudaStreamCreate(&stream);

        
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) throw std::runtime_error("Failed to read engine file!");
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);

        
        runtime.reset(createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        context.reset(engine->createExecutionContext());

        
        cudaMalloc(&buffers[0], inputSize);
        cudaMalloc(&buffers[1], outputSize);
    }

    ~TensorRT_OBB() {
        cudaFree(buffers[0]);
        cudaFree(buffers[1]);
        cudaStreamDestroy(stream);
    }

    void processFrame(cv::Mat& frame) {
      
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);

        // async memory copy from CPU to GPU
        cudaMemcpyAsync(buffers[0], blob.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream);

        context->enqueueV2(buffers, stream, nullptr);

        // again, async memory copy from CPU to GPU
        std::vector<float> output(1 * 20 * 8400);
        cudaMemcpyAsync(output.data(), buffers[1], outputSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

      
        std::vector<cv::RotatedRect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        float x_scale = (float)frame.cols / 640.0f;
        float y_scale = (float)frame.rows / 640.0f;

      
        for (int i = 0; i < 8400; ++i) {
            float cx = output[0 * 8400 + i] * x_scale;
            float cy = output[1 * 8400 + i] * y_scale;
            float w  = output[2 * 8400 + i] * x_scale;
            float h  = output[3 * 8400 + i] * y_scale;

            
            float maxScore = 0.0f;
            int classId = -1;
            for (int c = 4; c < 19; ++c) {
                float score = output[c * 8400 + i];
                if (score > maxScore) {
                    maxScore = score;
                    classId = c - 4;
                }
            }

            float angle = output[19 * 8400 + i] * (180.0f / CV_PI); // Convert radians to degrees

            if (maxScore > confThreshold) {
                
                boxes.push_back(cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(w, h), angle));
                confidences.push_back(maxScore);
                classIds.push_back(classId);
            }
        }

      
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        for (int idx : indices) {
            cv::RotatedRect rRect = boxes[idx];
            cv::Point2f vertices[4];
            rRect.points(vertices);

            // Draw the 4 corners
            for (int j = 0; j < 4; ++j) {
                cv::line(frame, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            
            std::string label = "Class " + std::to_string(classIds[idx]) + ": " + cv::format("%.2f", confidences[idx]);
            cv::putText(frame, label, vertices[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }
    }
};
//this is an OBB logic placeholder. Will put this part after assessing the performance of the latest ML model.
int main() {
    try {
        //to load our optimized TensorRT model compiled directly on Nano
        TensorRT_OBB model("detector.engine");

        cv::VideoCapture cap(0); 
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open USB Webcam /dev/video0" << std::endl;
            return -1;
        }

        cv::Mat frame;
        std::cout << "Starting C++ Real-Time Inference. Press ESC to exit." << std::endl;
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            model.processFrame(frame);

            cv::imshow("C++ TensorRT OBB Feed", frame);
            if (cv::waitKey(1) == 27) break;
        }

    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
