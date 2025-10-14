#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include "ObjectDetector.h"


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

private:
    Ui::MainWindow *ui;
    ObjectDetector detector;
    cv::Mat currentImage;
    cv::Mat resultImage;
    cv::VideoCapture videoCapture; // 新增：摄像头捕获对象
    QTimer *cameraTimer;           // 新增：定时器用于实时更新

    // 使用应用程序所在目录的路径
    QString weightsPath;
    QString configPath;

    bool isCameraRunning;  // 新增：摄像头状态标志

    void displayImage(const cv::Mat& image);
    void drawDetections(cv::Mat& image, const std::vector<DetectionResult>& detections);
    QImage cvMatToQImage(const cv::Mat& mat);
};

#endif // MAINWINDOW_H
