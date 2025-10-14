#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , weightsPath(QDir::cleanPath(QApplication::applicationDirPath() + "/../../models/yolov4.weights"))
    , configPath(QDir::cleanPath(QApplication::applicationDirPath() + "/../../models/yolov4.cfg"))
    , isCameraRunning(false)
{
    ui->setupUi(this);

    // 创建定时器
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &MainWindow::processCameraFrame);

    // 连接信号和槽
    connect(ui->loadButton, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(ui->detectButton, &QPushButton::clicked, this, &MainWindow::onDetectObjects);
    connect(ui->saveButton, &QPushButton::clicked, this, &MainWindow::onSaveResult);

    // 新增按钮连接（需要在UI中添加这些按钮）
    connect(ui->cameraStartButton, &QPushButton::clicked, this, &MainWindow::onCameraStart);
    connect(ui->cameraStopButton, &QPushButton::clicked, this, &MainWindow::onCameraStop);

    // 加载模型（需要下载YOLO模型文件）
    detector.loadModel(weightsPath.toStdString(), configPath.toStdString());

    setWindowTitle("物体识别应用 - 摄像头检测");
}

MainWindow::~MainWindow() {
    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
    delete ui;
}

void MainWindow::onCameraStart() {
    if (isCameraRunning) {
        return;
    }

    // 尝试打开摄像头（0通常是默认摄像头）
    if (!videoCapture.open(0)) {
        QMessageBox::warning(this, "错误", "无法打开摄像头");
        return;
    }

    // 设置摄像头参数（可选）
    videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    videoCapture.set(cv::CAP_PROP_FPS, 30);

    isCameraRunning = true;
    cameraTimer->start(33); // 约30fps，每33毫秒一帧

    ui->statusLabel->setText("摄像头运行中...");
}

void MainWindow::onCameraStop() {
    if (!isCameraRunning) {
        return;
    }

    cameraTimer->stop();
    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
    isCameraRunning = false;

    // 清空显示
    ui->imageLabel->clear();
    ui->imageLabel->setText("图像");
    ui->statusLabel->setText("摄像头已停止");
}

void MainWindow::processCameraFrame() {
    if (!videoCapture.isOpened()) {
        return;
    }

    cv::Mat frame;
    if (videoCapture.read(frame)) {
        if (!frame.empty()) {
            // 进行物体检测
            auto detections = detector.detect(frame);

            // 绘制检测结果
            cv::Mat resultFrame = frame.clone();
            drawDetections(resultFrame, detections);

            // 显示结果
            displayImage(resultFrame);

            // 更新状态
            QString status = QString("检测到 %1 个物体 | 摄像头运行中").arg(detections.size());
            ui->statusLabel->setText(status);
        }
    } else {
        QMessageBox::warning(this, "错误", "无法读取摄像头帧");
        onCameraStop();
    }
}

// 原有的其他函数保持不变
void MainWindow::onLoadImage() {
    // 如果摄像头在运行，先停止
    if (isCameraRunning) {
        onCameraStop();
    }

    QString fileName = QFileDialog::getOpenFileName(this,
                                                    "打开图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)");

    if (!fileName.isEmpty()) {
        currentImage = cv::imread(fileName.toStdString());
        if (!currentImage.empty()) {
            displayImage(currentImage);
            ui->statusLabel->setText("图像加载成功");
        } else {
            QMessageBox::warning(this, "错误", "无法加载图像文件");
        }
    }
}

void MainWindow::onDetectObjects() {
    // 如果摄像头在运行，使用当前帧进行检测
    if (isCameraRunning) {
        return; // 摄像头模式下自动持续检测
    }

    if (currentImage.empty()) {
        QMessageBox::warning(this, "错误", "请先加载图像或启动摄像头");
        return;
    }

    if (detector.loadModel(weightsPath.toStdString(), configPath.toStdString())) {
        auto detections = detector.detect(currentImage);
        resultImage = currentImage.clone();
        drawDetections(resultImage, detections);
        displayImage(resultImage);

        // 显示检测结果统计
        QString status = QString("检测到 %1 个物体").arg(detections.size());
        ui->statusLabel->setText(status);
    } else {
        QMessageBox::warning(this, "错误", "无法加载模型文件");
    }
}

void MainWindow::onSaveResult() {
    if (resultImage.empty() && !isCameraRunning) {
        QMessageBox::warning(this, "错误", "没有检测结果可保存");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this,
                                                    "保存结果", "", "图像文件 (*.png *.jpg *.jpeg)");

    if (!fileName.isEmpty()) {
        // 如果是摄像头模式，保存当前显示的帧
        if (isCameraRunning && videoCapture.isOpened()) {
            cv::Mat currentFrame;
            videoCapture.read(currentFrame);
            if (!currentFrame.empty()) {
                auto detections = detector.detect(currentFrame);
                drawDetections(currentFrame, detections);
                cv::imwrite(fileName.toStdString(), currentFrame);
            }
        } else {
            cv::imwrite(fileName.toStdString(), resultImage);
        }
        QMessageBox::information(this, "成功", "结果已保存");
    }
}

// 原有的displayImage、drawDetections、cvMatToQImage函数保持不变
void MainWindow::displayImage(const cv::Mat& image) {
    QImage qimage = cvMatToQImage(image);
    QPixmap pixmap = QPixmap::fromImage(qimage);

    // 缩放以适应标签大小
    pixmap = pixmap.scaled(ui->imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imageLabel->setPixmap(pixmap);
}

void MainWindow::drawDetections(cv::Mat& image, const std::vector<DetectionResult>& detections) {
    for (const auto& detection : detections) {
        cv::rectangle(image, detection.boundingBox, cv::Scalar(0, 255, 0), 2);

        std::string label = detection.className + ": " + std::to_string(detection.confidence).substr(0, 4);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(image,
                      cv::Point(detection.boundingBox.x, detection.boundingBox.y - labelSize.height - baseLine),
                      cv::Point(detection.boundingBox.x + labelSize.width, detection.boundingBox.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(image, label,
                    cv::Point(detection.boundingBox.x, detection.boundingBox.y - baseLine),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

QImage MainWindow::cvMatToQImage(const cv::Mat& mat) {
    if (mat.type() == CV_8UC1) {
        QImage image(mat.cols, mat.rows, QImage::Format_Grayscale8);
        image.bits();
        return image;
    } else if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    } else if (mat.type() == CV_8UC4) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32).copy();
    }
    return QImage();
}
