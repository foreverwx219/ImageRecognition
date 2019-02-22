#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    Mat image;
    QImage img;

private slots:
    void on_btn_readImg_clicked();

    void on_btn_ok_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
