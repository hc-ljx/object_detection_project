#pragma once

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QThread>
#include "ObjectDetector.h"
#include "FrameProcessor.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoadImage();
    void onDetectObjects();
    void onSaveResult();
    void onCameraStart();  // 新增：启动摄像头
    void onCameraStop();   // 新增：停止摄像头
    void processCameraFrame(); // 新增：处理摄像头帧
    void onFrameProcessed(const ProcessedFrame& result); // 新增：处理完成的帧

private:
    Ui::MainWindow *ui;
    ObjectDetector detector;
    cv::Mat currentImage;
    cv::Mat resultImage;
    cv::VideoCapture videoCapture;
    QTimer *cameraTimer;

    // 多线程处理
    QThread* processorThread;
    FrameProcessor* frameProcessor;
    cv::Mat lastProcessedFrame; // 最后处理完成的帧

    QString weightsPath;
    QString configPath;

    bool isCameraRunning;
    QMutex frameMutex;

    void displayImage(const cv::Mat& image);
    void drawDetections(cv::Mat& image, const std::vector<DetectionResult>& detections);
    QImage cvMatToQImage(const cv::Mat& mat);
};
