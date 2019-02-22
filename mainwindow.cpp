#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QString>
#include <QFileDialog>
#include <QMessageBox>
#include <opencv/cv.h>
#include <QTextCodec>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_readImg_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image File(*.bmp *.jpg *.jpeg *.png)"));
    QTextCodec * code = QTextCodec::codecForName("gb18030");
    std::string name = code->fromUnicode(filename).data();
    image = cv::imread(name);
    if(!image.data){
        QMessageBox msgBox;
        msgBox.setText(tr("image data is NULL"));
        msgBox.exec();
    }else{
        cv:cvtColor(image, image, CV_BGR2RGB);
        img = QImage((const unsigned char *)(image.data), image.cols, image.rows, image.step, QImage::Format_RGB888);
        this->ui->lb_src->clear();
        this->ui->lb_src->setPixmap(QPixmap::fromImage(img));
        //this->ui->lb_display->resize(this->ui->lb_display->pixmap()->size());
    }
}


void MainWindow::on_btn_ok_clicked()
{
    cv::Mat grayImg;
    cv::cvtColor(image, grayImg, CV_BGR2GRAY);
    if(this->ui->comboBox->currentIndex() == 0){
        img = QImage((const unsigned char*)(grayImg.data), grayImg.cols, grayImg.rows, grayImg.step, QImage::Format_Indexed8);
        this->ui->lb_display->setPixmap(QPixmap::fromImage(img));
    }else if(this->ui->comboBox->currentIndex() == 1){

    }else if(this->ui->comboBox->currentIndex() == 2){
        cv::Mat grad_x, grad_y, outImg;
        cv::Sobel(grayImg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
        convertScaleAbs(grad_x, grad_x);
        cv::Sobel(grayImg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
        convertScaleAbs(grad_y, grad_y);
        addWeighted(grad_x, 0.5, grad_y, 0.5, 0, outImg);
        img = QImage((const unsigned char*)(outImg.data), outImg.cols, outImg.rows, outImg.step, QImage::Format_Indexed8);
        this->ui->lb_display->setPixmap(QPixmap::fromImage(img));
    }else if(this->ui->comboBox->currentIndex() == 3){
        cv::Mat outImg;
        cv::Canny(grayImg, outImg, 3, 9, 3);
        img = QImage((const unsigned char*)(outImg.data), outImg.cols, outImg.rows, outImg.step, QImage::Format_Indexed8);
        this->ui->lb_display->setPixmap(QPixmap::fromImage(img));
    }else if(this->ui->comboBox->currentIndex() == 4){
        cv::Mat outImg;
        cv::blur(image, outImg, Size(7, 7));
        img = QImage((const unsigned char*)(outImg.data), outImg.cols, outImg.rows, outImg.step, QImage::Format_Indexed8);
        this->ui->lb_display->setPixmap(QPixmap::fromImage(img));
    }else if(this->ui->comboBox->currentIndex() == 5){
        cv::Mat outImg;
        cv::medianBlur(image, outImg, 7);
        img = QImage((const unsigned char*)(outImg.data), outImg.cols, outImg.rows, outImg.step, QImage::Format_Indexed8);
        this->ui->lb_display->setPixmap(QPixmap::fromImage(img));
    }else if(this->ui->comboBox->currentIndex() == 6){
        cv::Mat outImg;
        cv::GaussianBlur(image, outImg, Size(3, 3), 0, 0);
        img = QImage((const unsigned char*)(outImg.data), outImg.cols, outImg.rows, outImg.step, QImage::Format_Indexed8);
        this->ui->lb_display->setPixmap(QPixmap::fromImage(img));
    }else{
        cv::Mat outImg;
        cv::Mat element = cv::getStructuringElement(MORPH_RECT, Size(7, 1), Point(-1, -1));
        cv::erode(image, outImg, element);
        img = QImage((const unsigned char*)(outImg.data), outImg.cols, outImg.rows, outImg.step, QImage::Format_Indexed8);
        this->ui->lb_display->setPixmap(QPixmap::fromImage(img));

    }
}
