#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , weightsPath(QDir::cleanPath(QApplication::applicationDirPath() + "/../../models/yolov4.weights"))
    , configPath(QDir::cleanPath(QApplication::applicationDirPath() + "/../../models/yolov4.cfg"))
    , isCameraRunning(false)
    , cameraTimer(new QTimer(this))
    , processorThread(new QThread(this))
    , frameProcessor(new FrameProcessor())
{
    ui->setupUi(this);

    // 设置帧处理器
    frameProcessor->setDetector(&detector);
    frameProcessor->moveToThread(processorThread);

    // 连接信号和槽
    connect(cameraTimer, &QTimer::timeout, this, &MainWindow::processCameraFrame);
    connect(frameProcessor, &FrameProcessor::frameProcessed, this, &MainWindow::onFrameProcessed);
    connect(processorThread, &QThread::started, frameProcessor, &FrameProcessor::startProcessing);
    connect(processorThread, &QThread::finished, frameProcessor, &FrameProcessor::deleteLater);

    // 启动处理器线程
    processorThread->start();

    // 连接UI信号和槽
    connect(ui->loadButton, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(ui->detectButton, &QPushButton::clicked, this, &MainWindow::onDetectObjects);
    connect(ui->saveButton, &QPushButton::clicked, this, &MainWindow::onSaveResult);
    connect(ui->cameraStartButton, &QPushButton::clicked, this, &MainWindow::onCameraStart);
    connect(ui->cameraStopButton, &QPushButton::clicked, this, &MainWindow::onCameraStop);

    // 加载模型
    detector.loadModel(weightsPath.toStdString(), configPath.toStdString());

    setWindowTitle("物体识别应用 - 摄像头检测");
}

MainWindow::~MainWindow() {
    if (isCameraRunning) {
        onCameraStop();
    }

    // 停止处理器
    frameProcessor->stopProcessing();

    // 停止线程
    processorThread->quit();
    if (!processorThread->wait(2000)) { // 等待2秒
        processorThread->terminate(); // 强制终止
        processorThread->wait();
    }

    if (videoCapture.isOpened()) {
        videoCapture.release();
    }
    delete ui;
}

void MainWindow::onCameraStart() {
    if (isCameraRunning) {
        return;
    }

    if (!videoCapture.open(0)) {
        QMessageBox::warning(this, "错误", "无法打开摄像头");
        return;
    }

    // 使用适中的分辨率
    videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 480);
    videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    videoCapture.set(cv::CAP_PROP_FPS, 15);

    isCameraRunning = true;

    // 清空之前的帧
    {
        QMutexLocker locker(&frameMutex);
        lastProcessedFrame = cv::Mat();
    }

    cameraTimer->start(66); // 15fps的采集，给处理留出时间

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
            // 发送到工作线程处理
            frameProcessor->processFrame(frame);
        }
    } else {
        QMessageBox::warning(this, "错误", "无法读取摄像头帧");
        onCameraStop();
    }
}

void MainWindow::onFrameProcessed(const ProcessedFrame& result) {
    if (result.isValid) {
        QMutexLocker locker(&frameMutex);

        // 保存处理后的帧
        lastProcessedFrame = result.frame.clone();
        drawDetections(lastProcessedFrame, result.detections);

        // 统一在这里显示帧
        displayImage(lastProcessedFrame);

        // 更新状态
        QString status = QString("检测到 %1 个物体 | 摄像头运行中").arg(result.detections.size());
        ui->statusLabel->setText(status);
    } else {
        // 如果处理失败，至少显示原始帧保持流畅
        QMutexLocker locker(&frameMutex);
        if (!lastProcessedFrame.empty()) {
            displayImage(lastProcessedFrame);
        }
    }
}

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
        if (isCameraRunning) {
            QMutexLocker locker(&frameMutex);
            if (!lastProcessedFrame.empty()) {
                cv::imwrite(fileName.toStdString(), lastProcessedFrame);
            }
        } else {
            cv::imwrite(fileName.toStdString(), resultImage);
        }
        QMessageBox::information(this, "成功", "结果已保存");
    }
}

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
