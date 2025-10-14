#include "ObjectDetector.h"
#include <fstream>
#include<iostream>

ObjectDetector::ObjectDetector() : confidenceThreshold(0.5f), nmsThreshold(0.4f) {
    // 初始化类名（COCO数据集）
    classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}

bool ObjectDetector::loadModel(const std::string& modelPath, const std::string& configPath) {
    try {
        // 检查文件是否存在
        std::ifstream modelFile(modelPath);
        std::ifstream configFile(configPath);

        if (!modelFile.good()) {
            std::cerr << "模型权重文件不存在: " << modelPath << std::endl;
            return false;
        }
        if (!configFile.good()) {
            std::cerr << "配置文件不存在: " << configPath << std::endl;
            return false;
        }

        if (configPath.empty()) {
            net = cv::dnn::readNet(modelPath);
        } else {
            net = cv::dnn::readNet(modelPath, configPath);
        }

        if (net.empty()) {
            std::cerr << "无法加载网络模型" << std::endl;
            return false;
        }

        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::cout << "模型加载成功" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载模型时发生异常: " << e.what() << std::endl;
        return false;
    }
}

std::vector<DetectionResult> ObjectDetector::detect(cv::Mat& image) {
    std::vector<DetectionResult> results;

    if (net.empty()) {
        return results;
    }

    // 准备输入blob
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(416, 416), cv::Scalar(0,0,0), true, false);
    net.setInput(blob);

    // 前向传播
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames());

    // 处理输出
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& output : outs) {
        auto* data = (float*)output.data;
        for (int i = 0; i < output.rows; ++i, data += output.cols) {
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confidenceThreshold) {
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // 应用非极大值抑制
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    for (int idx : indices) {
        DetectionResult result;
        result.boundingBox = boxes[idx];
        result.className = classNames[classIds[idx]];
        result.confidence = confidences[idx];
        results.push_back(result);
    }

    return results;
}

void ObjectDetector::setConfidenceThreshold(float threshold) {
    confidenceThreshold = threshold;
}

void ObjectDetector::setNMSThreshold(float threshold) {
    nmsThreshold = threshold;
}

std::vector<std::string> ObjectDetector::getOutputsNames() {
    static std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}
