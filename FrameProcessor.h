#pragma once

#include <QObject>
#include <QImage>
#include <QPixmap>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QAtomicInteger>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.h"

struct ProcessedFrame {
    cv::Mat frame;
    std::vector<DetectionResult> detections;
    bool isValid;
};

class FrameProcessor : public QObject
{
    Q_OBJECT

public:
    explicit FrameProcessor(QObject *parent = nullptr);
    ~FrameProcessor();

    void setDetector(ObjectDetector* detector);
    void processFrame(const cv::Mat& frame);
    void stopProcessing();

signals:
    void frameProcessed(const ProcessedFrame& result);

public slots:
    void startProcessing();

private:
    ObjectDetector* m_detector;
    QMutex m_mutex;
    QWaitCondition m_condition;
    std::vector<cv::Mat> m_frameQueue;
    bool m_stopRequested;
    bool m_isProcessing;
};


