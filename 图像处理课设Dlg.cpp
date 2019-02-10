
// 图像处理课设Dlg.cpp : 实现文件
//

#include "stdafx.h"
#include "图像处理课设.h"
#include "图像处理课设Dlg.h"
#include "afxdialogex.h"
#include <opencv2\core\core.hpp>  
#include <opencv2\opencv.hpp>
#include <opencv\highgui.h>
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace cv;
// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

Mat srcimg;
Mat greyimg;

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// C图像处理课设Dlg 对话框



C图像处理课设Dlg::C图像处理课设Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MY_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void C图像处理课设Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(C图像处理课设Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &C图像处理课设Dlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON1, &C图像处理课设Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &C图像处理课设Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &C图像处理课设Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &C图像处理课设Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &C图像处理课设Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &C图像处理课设Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &C图像处理课设Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &C图像处理课设Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &C图像处理课设Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &C图像处理课设Dlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &C图像处理课设Dlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON12, &C图像处理课设Dlg::OnBnClickedButton12)
	ON_BN_CLICKED(IDC_BUTTON13, &C图像处理课设Dlg::OnBnClickedButton13)
	ON_BN_CLICKED(IDC_BUTTON14, &C图像处理课设Dlg::OnBnClickedButton14)
	ON_BN_CLICKED(IDC_BUTTON15, &C图像处理课设Dlg::OnBnClickedButton15)
	ON_BN_CLICKED(IDC_BUTTON16, &C图像处理课设Dlg::OnBnClickedButton16)
	ON_BN_CLICKED(IDC_BUTTON17, &C图像处理课设Dlg::OnBnClickedButton17)
END_MESSAGE_MAP()


// C图像处理课设Dlg 消息处理程序

BOOL C图像处理课设Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void C图像处理课设Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void C图像处理课设Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR C图像处理课设Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void C图像处理课设Dlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}


void C图像处理课设Dlg::OnBnClickedButton1()
{
	//读入一张图片  
	//srcimg= imread("C://Users//bettertime//Desktop//1.jpg");
	srcimg = imread("Test1.jpg");
	// 在窗口中显示原画    
	imshow("原图", srcimg);
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton2()
{
	if (srcimg.data)  //判断是否有数据
	{
		cvtColor(srcimg, greyimg, CV_BGR2GRAY);
		imshow("灰度图", greyimg);
	}
	// TODO: 在此添加控件通知处理程序代码
}



void C图像处理课设Dlg::OnBnClickedButton3()
{	
	if (greyimg.data)  //判断是否有数据
	{
		// 需要计算图像的哪个通道（bgr空间需要确定计算 b或g或r空间）
		const int channels[1] = { 0 };

		//直方图的每一个维度的 柱条的数目（就是将灰度级分组）  
		int histSize[] = { 256 };   //如果这里写成int histSize = 256;   那么下面调用计算直方图的函数的时候，该变量需要写 &histSize  

									//定义一个变量用来存储 单个维度 的数值的取值范围    
		float midRanges[] = { 0, 256 };

		//确定每个维度的取值范围，就是横坐标的总数    
		const float *ranges[] = { midRanges };

		//输出的结果存储的 空间 ，用MatND类型来存储结果  
		MatND dstHist;

		calcHist(&greyimg, 1, channels, Mat(), dstHist, 1, histSize, ranges, true, false);

		//calcHist  函数调用结束后，dstHist变量中将储存了 直方图的信息  用dstHist的模版函数 at<Type>(i)得到第i个柱条的值  at<Type>(i, j)得到第i个并且第j个柱条的值    

		//首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像    
		Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);

		//一个图像的某个灰度级的像素个数（最多为图像像素总数），可能会超过显示直方图的所定义的图像的尺寸，因此绘制直方图的时候，让直方图最高的地方只有图像高度的90%来显示  

		//先用minMaxLoc函数来得到计算直方图后的像素的最大个数    
		double g_dHistMaxValue;
		minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);

		//遍历直方图得到的数据    
		for (int i = 0; i < 256; i++)
		{
			int value = cvRound(256 * 0.9 *(dstHist.at<float>(i) / g_dHistMaxValue));

			line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));
		}
		imshow("绘制灰度直方图", drawImage);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton4()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		bilateralFilter(srcimg, out, 5, 10, 10);//双边滤波
		Mat imageEnhance;
		Mat  kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
		filter2D(out, out, CV_8UC3, kernel);
		imshow("拉普拉斯锐化", out);

	}
	
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton5()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		Canny(srcimg, out, 3, 9, 3);
		imshow("canny边缘", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton6()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y, out;
		//求x方向梯度
		Sobel(srcimg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		//求y方向梯度
		Sobel(srcimg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		//合并梯度
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out);
		imshow("soble", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton7()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		blur(srcimg, out, Size(7, 7));
		imshow("均值滤波", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton8()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		medianBlur(srcimg, out, 7);
		imshow("中值滤波", out);
	}

	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton9()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		GaussianBlur(srcimg, out, Size(3,3),0,0);
		imshow("高斯滤波", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton10()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 1), Point(-1, -1));
		erode(srcimg, out, element);
		imshow("腐蚀滤波", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton11()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		Point2f AffinePoints0[3] = { Point2f(100, 50), Point2f(100, 390), Point2f(600, 50) };
		Point2f AffinePoints1[3] = { Point2f(200, 100), Point2f(200, 330), Point2f(500, 50) };
		Mat Trans = getAffineTransform(AffinePoints0, AffinePoints1);
		warpAffine(srcimg, out, Trans, Size(srcimg.cols, srcimg.rows));
		imshow("仿射变换", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton12()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat out;
		Point2f AffinePoints0[4] = { Point2f(100, 50), Point2f(100, 390), Point2f(600, 50), Point2f(600, 390) };
		Point2f AffinePoints1[4] = { Point2f(200, 100), Point2f(200, 330), Point2f(500, 50), Point2f(600, 390) };
		Mat Trans = getPerspectiveTransform(AffinePoints0, AffinePoints1);
		warpPerspective(srcimg, out, Trans, Size(srcimg.cols, srcimg.rows), CV_INTER_CUBIC);
		imshow("投影变换", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton13()
{
	if (greyimg.data)  //判断是否有数据
	{
		Mat out;
		adaptiveThreshold(greyimg, out, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);
		imshow("自适应阙值化", out);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton14()
{
	if (srcimg.data)  //判断是否有数据
	{
		Mat img, templ, result;
		img = srcimg;
		templ = imread("Test2.jpg");

		int result_cols = img.cols - templ.cols + 1;
		int result_rows = img.rows - templ.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(img, templ, result, CV_TM_SQDIFF_NORMED);//这里我们使用的匹配算法是标准平方差匹配 method=CV_TM_SQDIFF_NORMED，数值越小匹配度越好
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal = -1;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		matchLoc = minLoc;
		rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);
		imshow("模板匹配", img);
	}
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton15()
{
	Mat image01 = imread("Test3.png", 1);    //右图
	Mat image02 = imread("Test4.png", 1);    //左图
	imshow("pic3", image01);
	imshow("pic4", image02);

	//灰度图转换  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);


	//提取特征点    
	SurfFeatureDetector surfDetector(800);  // 海塞矩阵阈值，在这里调整精度，值越大点越少，越精准 
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(image1, keyPoint1);
	surfDetector.detect(image2, keyPoint2);

	//特征点描述，为下边的特征点匹配做准备    
	SurfDescriptorExtractor SurfDescriptor;
	Mat imageDesc1, imageDesc2;
	SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
	SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

	//获得匹配特征点，并提取最优配对     
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;

	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	Mat img_match;
	drawMatches(image01, keyPoint1, image02, keyPoint2, matchePoints, img_match);
	imshow("match", img_match);
	waitKey(0);
	// TODO: 在此添加控件通知处理程序代码
}


void C图像处理课设Dlg::OnBnClickedButton16()
{
	Mat obj = imread("Test3.png");   //前帧
	Mat scene = imread("Test4.png"); //当前帧
	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	ORB detector;     //采用ORB算法提取特征点
	detector.detect(obj, obj_keypoints);//得到前帧特征点
	detector.detect(scene, scene_keypoints);//得到当前帧特征点
	detector.compute(obj, obj_keypoints, obj_descriptors);//计算描述符（特征向量）
	detector.compute(scene, scene_keypoints, scene_descriptors);
	BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量
	vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches);
	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img); //源图像1的特征点匹配源图像2的特征点[matches[i]]
	imshow("滤除误匹配前", match_img);

	//保存匹配对序号
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;//此匹配对应的查询图像的特征描述子索引，可以据此找到匹配点（前帧特征点）在查询图像中的位置
		trainIdxs[i] = matches[i].trainIdx;//此匹配对应的训练(模板)图像的特征描述子索引，可以据此找到匹配点（前帧特征点）在训练图像中的位置
	}

	Mat H12;   //变换矩阵

	vector<Point2f> points1; KeyPoint::convert(obj_keypoints, points1, queryIdxs);//转移到points？
	vector<Point2f> points2; KeyPoint::convert(scene_keypoints, points2, trainIdxs);
	int ransacReprojThreshold = 5;  //拒绝阈值


	H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);//通过两帧的特征点寻找透视变换矩阵 ransacReprojThreshold即通过基于ransac的鲁棒性方法
	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);//
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
		{
			matchesMask[i1] = 1;//决定哪些点被画出
		}
	}
	Mat match_img2;   //滤除‘外点’后
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);// Scalar::all(-1)点颜色随机 matchesMask决定哪些点被画出

																																  //画出目标位置
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(obj.cols, 0);
	obj_corners[2] = cvPoint(obj.cols, obj.rows); obj_corners[3] = cvPoint(0, obj.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H12);
	line(match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);

	imshow("滤除误匹配后", match_img2);
	waitKey(0);
	// TODO: 在此添加控件通知处理程序代码
}

// 用HSV中的Hue分量进行跟踪
IplImage *image = 0, *hsv = 0, *hue = 0, *mask = 0, *backproject = 0, *histimg = 0;
CvHistogram *hist = 0;          //直方图类
bool gotBB = false;//目标选取标志位
int backproject_mode = 0;
int select_object = 0;
int track_object = 0;
int show_hist = 1;
CvPoint origin;
CvRect selection;
CvRect track_window;
CvBox2D track_box;              //Meanshift跟踪算法返回的Box类
CvConnectedComp track_comp;
int hdims = 16;                 //划分直方图bins的个数，越多越精确
float hranges_arr[] = { 0,180 };  //像素值的范围
float* hranges = hranges_arr;   //用于初始化CvHistogram类
int vmin = 10, vmax = 65, smin = 30;

void on_mouse(int event, int x, int y, int flags, void* param) //该函数用于选择跟踪目标
{
	if (!image)
		return;

	if (image->origin)
		y = image->height - y;

	if (select_object) //如果处于选择跟踪物体阶段，则对selection用当前的鼠标位置进行设置
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = selection.x + CV_IABS(x - origin.x);
		selection.height = selection.y + CV_IABS(y - origin.y);

		selection.x = MAX(selection.x, 0);
		selection.y = MAX(selection.y, 0);
		selection.width = MIN(selection.width, image->width);
		selection.height = MIN(selection.height, image->height);
		selection.width -= selection.x;
		selection.height -= selection.y;
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:        //开始点击选择跟踪物体
		origin = cvPoint(x, y);
		selection = cvRect(x, y, 0, 0);  //坐标
		select_object = 1;            //表明开始进行选取
		break;
	case CV_EVENT_LBUTTONUP:
		select_object = 0;            //选取完成
		if (selection.width > 0 && selection.height > 0)
			track_object = -1;        //如果选择物体有效，则打开跟踪功能
		gotBB = true;
		break;
	}
}

CvScalar hsv2rgb(float hue)         //用于将Hue量转换成RGB量
{
	int rgb[3], p, sector;
	static const int sector_data[][3] =
	{ { 0,2,1 },{ 1,2,0 },{ 1,0,2 },{ 2,0,1 },{ 2,1,0 },{ 0,1,2 } };
	hue *= 0.033333333333333333333333333333333f;
	sector = cvFloor(hue);
	p = cvRound(255 * (hue - sector));
	p ^= sector & 1 ? 255 : 0;

	rgb[sector_data[sector][0]] = 255;
	rgb[sector_data[sector][1]] = 0;
	rgb[sector_data[sector][2]] = p;

	return cvScalar(rgb[2], rgb[1], rgb[0], 0);  //返回对应的颜色值
}

void C图像处理课设Dlg::OnBnClickedButton17()
{
	CvCapture* capture = 0;

	capture = cvCaptureFromAVI("plane.avi"); //打开AVI文件

	if (!capture)  //打开视频流失败处理
	{
		fprintf(stderr, "Could not initialize capturing...\n");
		return;
	}
	//打印出程序功能列表
	printf("Hot keys: \n"
		"\tESC - quit the program\n"
		"\tc - stop the tracking\n"
		"\tb - switch to/from backprojection view\n"
		"\th - show/hide object histogram\n"
		"To initialize tracking, select the object with mouse\n");

	cvNamedWindow("Histogram", 1);
	cvNamedWindow("CamShiftDemo", 1);
	cvSetMouseCallback("CamShiftDemo", on_mouse, 0);        // 设置鼠标回调函数
	cvCreateTrackbar("Vmin", "CamShiftDemo", &vmin, 256, 0);//建立滑动条
	cvCreateTrackbar("Vmax", "CamShiftDemo", &vmax, 256, 0);
	cvCreateTrackbar("Smin", "CamShiftDemo", &smin, 256, 0);

	//cvSetMouseCallback("CamShiftDemo", on_mouse, NULL ); 

	IplImage* frame = 0;
	int frameCount = 0;
	for (frameCount = 0; frameCount<5; frameCount++)
	{
		frame = cvQueryFrame(capture);
	}

	bool fromfile = true;

	if (!image)  //刚开始先建立一些缓冲区
	{
		/* allocate all the buffers */
		image = cvCreateImage(cvGetSize(frame), 8, 3);
		image->origin = frame->origin;
		hsv = cvCreateImage(cvGetSize(frame), 8, 3);
		hue = cvCreateImage(cvGetSize(frame), 8, 1);
		mask = cvCreateImage(cvGetSize(frame), 8, 1);       //分配掩膜图像空间
		backproject = cvCreateImage(cvGetSize(frame), 8, 1);//分配反向投影图空间，大小一样，单通道
		hist = cvCreateHist(1, &hdims, CV_HIST_ARRAY, &hranges, 1);  //分配建立直方图空间
		histimg = cvCreateImage(cvSize(320, 200), 8, 3);              //分配用于画直方图的空间
		cvZero(histimg);   //背景为黑色
	}

	while (!gotBB)
	{
		if (!fromfile)
			frame = cvQueryFrame(capture);

		cvShowImage("CamShiftDemo", frame);
		if (cvWaitKey(20) == 27)
			return;
	}

	////imshow("Tracker", frame);
	////Remove callback
	//cvSetMouseCallback("CamShiftDemo", NULL, NULL ); 


	for (;;)           //进入视频帧处理主循环
	{
		int i, bin_w, c;

		frame = cvQueryFrame(capture);
		if (!frame)
			break;


		cvCopy(frame, image, 0);
		cvCvtColor(image, hsv, CV_BGR2HSV);   // 把图像从RGB表色系转为HSV表色系

		if (track_object)  //   如果当前有需要跟踪的物体   
		{
			int _vmin = vmin, _vmax = vmax;

			//掩膜板，只处理像素值为H：0~180，S：smin~256，V：vmin~vmax之间的部分
			void cvInRangeS(const CvArr* src, CvScalar lower, CvScalar upper, CvArr* dst);
			cvInRangeS(hsv, cvScalar(0, smin, MIN(_vmin, _vmax), 0),
				cvScalar(180, 256, MAX(_vmin, _vmax), 0), mask);
			cvSplit(hsv, hue, 0, 0, 0); // 取得H分量

			if (track_object < 0) //如果需要跟踪的物体还没有进行属性提取，则进行选取框类的图像属性提取
			{
				float max_val = 0.f;
				cvSetImageROI(hue, selection);  // 设置原选择框
				cvSetImageROI(mask, selection); // 设置Mask的选择框
				cvCalcHist(&hue, hist, 0, mask);// 得到选择框内且满足掩膜板内的直方图
				cvGetMinMaxHistValue(hist, 0, &max_val, 0, 0);
				cvConvertScale(hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0);  // 对直方图转为0~255
				cvResetImageROI(hue);  // remove ROI
				cvResetImageROI(mask);
				track_window = selection;
				track_object = 1;

				cvZero(histimg);
				bin_w = histimg->width / hdims;
				for (i = 0; i < hdims; i++)  //画直方图到图像空间
				{
					int val = cvRound(cvGetReal1D(hist->bins, i)*histimg->height / 255);
					//对一个double型的数进行四舍五入，并返回一个整型数
					CvScalar color = hsv2rgb(i*180.f / hdims);
					cvRectangle(histimg, cvPoint(i*bin_w, histimg->height),
						cvPoint((i + 1)*bin_w, histimg->height - val),
						color, -1, 8, 0);
				}
			}

			// 得到hue的反向投影图           
			cvCalcBackProject(&hue, backproject, hist);
			// 得到反向投影图mask内的内容
			cvAnd(backproject, mask, backproject, 0);
			cvCamShift(backproject, track_window,
				cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1),
				&track_comp, &track_box);
			//使用MeanShift算法对backproject中的内容进行搜索，返回跟踪结果

			track_window = track_comp.rect; //得到跟踪结果的矩形框

			if (backproject_mode)
				cvCvtColor(backproject, image, CV_GRAY2BGR); // 显示模式
			if (!image->origin)
				track_box.angle = -track_box.angle;
			cvEllipseBox(image, track_box, CV_RGB(255, 0, 0), 3, CV_AA, 0); //画出跟踪结果的位置
		}

		if (select_object && selection.width > 0 && selection.height > 0) //如果正处于物体选择，画出选择框
		{
			cvSetImageROI(image, selection);
			cvXorS(image, cvScalarAll(255), image, 0);
			cvResetImageROI(image);
		}

		cvShowImage("CamShiftDemo", image); //显示视频和直方图
		cvShowImage("Histogram", histimg);

		c = cvWaitKey(10);
		if ((char)c == 27)
			break;
		switch ((char)c)
		{
		case 'b':
			backproject_mode ^= 1;
			break;
		case 'c':
			track_object = 0;
			cvZero(histimg);
			break;
		case 'h':
			show_hist ^= 1;
			if (!show_hist)
				cvDestroyWindow("Histogram");
			else
				cvNamedWindow("Histogram", 1);
			break;
		default:
			;
		}
	}

	cvReleaseCapture(&capture);
	cvDestroyWindow("CamShiftDemo");


	// TODO: 在此添加控件通知处理程序代码
}
