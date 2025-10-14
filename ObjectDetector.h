#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct DetectionResult {
    cv::Rect boundingBox;
    std::string className;
    float confidence;
};

class ObjectDetector {
public:
    ObjectDetector();
    bool loadModel(const std::string& modelPath, const std::string& configPath = "");
    std::vector<DetectionResult> detect(cv::Mat& image);
    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);

private:
    cv::dnn::Net net;
    float confidenceThreshold;
    float nmsThreshold;
    std::vector<std::string> classNames;

    void loadClassNames();
    std::vector<std::string> getOutputsNames();
};

#endif // OBJECTDETECTOR_H
