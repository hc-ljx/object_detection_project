#include "FrameProcessor.h"
#include <QDebug>

FrameProcessor::FrameProcessor(QObject *parent)
    : QObject(parent)
    , m_detector(nullptr)
    , m_stopRequested(false)
    , m_isProcessing(false)
{
}

FrameProcessor::~FrameProcessor()
{
    stopProcessing();
}

void FrameProcessor::setDetector(ObjectDetector* detector)
{
    QMutexLocker locker(&m_mutex);
    m_detector = detector;
}

void FrameProcessor::processFrame(const cv::Mat& frame)
{
    if (!m_isProcessing) {
        return;
    }

    QMutexLocker locker(&m_mutex);

    // 使用双缓冲策略：只保留最新的一帧
    if (m_frameQueue.size() >= 1) {
        m_frameQueue.clear(); // 清空旧帧，只处理最新帧
    }
    m_frameQueue.push_back(frame.clone());
    m_condition.wakeOne();
}

void FrameProcessor::stopProcessing()
{
    m_stopRequested = true;  // 直接赋值
    m_isProcessing = false;  // 直接赋值

    {
        QMutexLocker locker(&m_mutex);
        m_condition.wakeAll();
    }
}

void FrameProcessor::startProcessing()
{
    m_stopRequested = false;
    m_isProcessing = true;

    // 记录帧时间用于平滑处理
    auto lastProcessTime = std::chrono::steady_clock::now();

    while (!m_stopRequested) {
        cv::Mat frame;
        bool hasFrame = false;

        {
            QMutexLocker locker(&m_mutex);

            if (!m_frameQueue.empty()) {
                frame = m_frameQueue.front();
                m_frameQueue.clear();
                hasFrame = true;
            } else {
                // 没有帧时等待较短时间
                m_condition.wait(&m_mutex, 16); // 约60fps的等待时间
                continue;
            }
        }

        if (hasFrame && !frame.empty()) {
            // 处理帧
            ProcessedFrame result;
            result.isValid = false;

            if (m_detector) {
                try {
                    auto detections = m_detector->detect(frame);
                    result.frame = frame;
                    result.detections = detections;
                    result.isValid = true;

                    emit frameProcessed(result);
                } catch (const std::exception& e) {
                    qWarning() << "Frame processing error:" << e.what();
                }
            }

            // 计算处理时间并适当延迟，保持稳定帧率
            auto currentTime = std::chrono::steady_clock::now();
            auto processDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTime - lastProcessTime);

            // 目标帧率：15fps ≈ 66ms/帧
            const int targetFrameTime = 66; // ms
            if (processDuration.count() < targetFrameTime) {
                QThread::msleep(targetFrameTime - processDuration.count());
            }

            lastProcessTime = std::chrono::steady_clock::now();
        }
    }

    // 清理
    QMutexLocker locker(&m_mutex);
    m_frameQueue.clear();
    m_isProcessing = false;
    qDebug() << "Frame processor stopped";
}
