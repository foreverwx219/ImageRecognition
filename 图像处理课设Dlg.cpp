
// ͼ�������Dlg.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "ͼ�������.h"
#include "ͼ�������Dlg.h"
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
// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

Mat srcimg;
Mat greyimg;

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
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


// Cͼ�������Dlg �Ի���



Cͼ�������Dlg::Cͼ�������Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MY_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void Cͼ�������Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(Cͼ�������Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &Cͼ�������Dlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON1, &Cͼ�������Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &Cͼ�������Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &Cͼ�������Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &Cͼ�������Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &Cͼ�������Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &Cͼ�������Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &Cͼ�������Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &Cͼ�������Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &Cͼ�������Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &Cͼ�������Dlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &Cͼ�������Dlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON12, &Cͼ�������Dlg::OnBnClickedButton12)
	ON_BN_CLICKED(IDC_BUTTON13, &Cͼ�������Dlg::OnBnClickedButton13)
	ON_BN_CLICKED(IDC_BUTTON14, &Cͼ�������Dlg::OnBnClickedButton14)
	ON_BN_CLICKED(IDC_BUTTON15, &Cͼ�������Dlg::OnBnClickedButton15)
	ON_BN_CLICKED(IDC_BUTTON16, &Cͼ�������Dlg::OnBnClickedButton16)
	ON_BN_CLICKED(IDC_BUTTON17, &Cͼ�������Dlg::OnBnClickedButton17)
END_MESSAGE_MAP()


// Cͼ�������Dlg ��Ϣ�������

BOOL Cͼ�������Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
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

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO: �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

void Cͼ�������Dlg::OnSysCommand(UINT nID, LPARAM lParam)
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

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void Cͼ�������Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR Cͼ�������Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void Cͼ�������Dlg::OnBnClickedOk()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CDialogEx::OnOK();
}


void Cͼ�������Dlg::OnBnClickedButton1()
{
	//����һ��ͼƬ  
	//srcimg= imread("C://Users//bettertime//Desktop//1.jpg");
	srcimg = imread("Test1.jpg");
	// �ڴ�������ʾԭ��    
	imshow("ԭͼ", srcimg);
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton2()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		cvtColor(srcimg, greyimg, CV_BGR2GRAY);
		imshow("�Ҷ�ͼ", greyimg);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}



void Cͼ�������Dlg::OnBnClickedButton3()
{	
	if (greyimg.data)  //�ж��Ƿ�������
	{
		// ��Ҫ����ͼ����ĸ�ͨ����bgr�ռ���Ҫȷ������ b��g��r�ռ䣩
		const int channels[1] = { 0 };

		//ֱ��ͼ��ÿһ��ά�ȵ� ��������Ŀ�����ǽ��Ҷȼ����飩  
		int histSize[] = { 256 };   //�������д��int histSize = 256;   ��ô������ü���ֱ��ͼ�ĺ�����ʱ�򣬸ñ�����Ҫд &histSize  

									//����һ�����������洢 ����ά�� ����ֵ��ȡֵ��Χ    
		float midRanges[] = { 0, 256 };

		//ȷ��ÿ��ά�ȵ�ȡֵ��Χ�����Ǻ����������    
		const float *ranges[] = { midRanges };

		//����Ľ���洢�� �ռ� ����MatND�������洢���  
		MatND dstHist;

		calcHist(&greyimg, 1, channels, Mat(), dstHist, 1, histSize, ranges, true, false);

		//calcHist  �������ý�����dstHist�����н������� ֱ��ͼ����Ϣ  ��dstHist��ģ�溯�� at<Type>(i)�õ���i��������ֵ  at<Type>(i, j)�õ���i�����ҵ�j��������ֵ    

		//�����ȴ���һ���ڵ׵�ͼ��Ϊ�˿�����ʾ��ɫ�����Ըû���ͼ����һ��8λ��3ͨ��ͼ��    
		Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);

		//һ��ͼ���ĳ���Ҷȼ������ظ��������Ϊͼ�����������������ܻᳬ����ʾֱ��ͼ���������ͼ��ĳߴ磬��˻���ֱ��ͼ��ʱ����ֱ��ͼ��ߵĵط�ֻ��ͼ��߶ȵ�90%����ʾ  

		//����minMaxLoc�������õ�����ֱ��ͼ������ص�������    
		double g_dHistMaxValue;
		minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);

		//����ֱ��ͼ�õ�������    
		for (int i = 0; i < 256; i++)
		{
			int value = cvRound(256 * 0.9 *(dstHist.at<float>(i) / g_dHistMaxValue));

			line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0));
		}
		imshow("���ƻҶ�ֱ��ͼ", drawImage);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton4()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		bilateralFilter(srcimg, out, 5, 10, 10);//˫���˲�
		Mat imageEnhance;
		Mat  kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
		filter2D(out, out, CV_8UC3, kernel);
		imshow("������˹��", out);

	}
	
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton5()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		Canny(srcimg, out, 3, 9, 3);
		imshow("canny��Ե", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton6()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y, out;
		//��x�����ݶ�
		Sobel(srcimg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		//��y�����ݶ�
		Sobel(srcimg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		//�ϲ��ݶ�
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out);
		imshow("soble", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton7()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		blur(srcimg, out, Size(7, 7));
		imshow("��ֵ�˲�", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton8()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		medianBlur(srcimg, out, 7);
		imshow("��ֵ�˲�", out);
	}

	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton9()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		GaussianBlur(srcimg, out, Size(3,3),0,0);
		imshow("��˹�˲�", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton10()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 1), Point(-1, -1));
		erode(srcimg, out, element);
		imshow("��ʴ�˲�", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton11()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		Point2f AffinePoints0[3] = { Point2f(100, 50), Point2f(100, 390), Point2f(600, 50) };
		Point2f AffinePoints1[3] = { Point2f(200, 100), Point2f(200, 330), Point2f(500, 50) };
		Mat Trans = getAffineTransform(AffinePoints0, AffinePoints1);
		warpAffine(srcimg, out, Trans, Size(srcimg.cols, srcimg.rows));
		imshow("����任", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton12()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		Point2f AffinePoints0[4] = { Point2f(100, 50), Point2f(100, 390), Point2f(600, 50), Point2f(600, 390) };
		Point2f AffinePoints1[4] = { Point2f(200, 100), Point2f(200, 330), Point2f(500, 50), Point2f(600, 390) };
		Mat Trans = getPerspectiveTransform(AffinePoints0, AffinePoints1);
		warpPerspective(srcimg, out, Trans, Size(srcimg.cols, srcimg.rows), CV_INTER_CUBIC);
		imshow("ͶӰ�任", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton13()
{
	if (greyimg.data)  //�ж��Ƿ�������
	{
		Mat out;
		adaptiveThreshold(greyimg, out, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);
		imshow("����Ӧ��ֵ��", out);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton14()
{
	if (srcimg.data)  //�ж��Ƿ�������
	{
		Mat img, templ, result;
		img = srcimg;
		templ = imread("Test2.jpg");

		int result_cols = img.cols - templ.cols + 1;
		int result_rows = img.rows - templ.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(img, templ, result, CV_TM_SQDIFF_NORMED);//��������ʹ�õ�ƥ���㷨�Ǳ�׼ƽ����ƥ�� method=CV_TM_SQDIFF_NORMED����ֵԽСƥ���Խ��
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal = -1;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		matchLoc = minLoc;
		rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);
		imshow("ģ��ƥ��", img);
	}
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton15()
{
	Mat image01 = imread("Test3.png", 1);    //��ͼ
	Mat image02 = imread("Test4.png", 1);    //��ͼ
	imshow("pic3", image01);
	imshow("pic4", image02);

	//�Ҷ�ͼת��  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);


	//��ȡ������    
	SurfFeatureDetector surfDetector(800);  // ����������ֵ��������������ȣ�ֵԽ���Խ�٣�Խ��׼ 
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(image1, keyPoint1);
	surfDetector.detect(image2, keyPoint2);

	//������������Ϊ�±ߵ�������ƥ����׼��    
	SurfDescriptorExtractor SurfDescriptor;
	Mat imageDesc1, imageDesc2;
	SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
	SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

	//���ƥ�������㣬����ȡ�������     
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;

	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	Mat img_match;
	drawMatches(image01, keyPoint1, image02, keyPoint2, matchePoints, img_match);
	imshow("match", img_match);
	waitKey(0);
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void Cͼ�������Dlg::OnBnClickedButton16()
{
	Mat obj = imread("Test3.png");   //ǰ֡
	Mat scene = imread("Test4.png"); //��ǰ֡
	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	ORB detector;     //����ORB�㷨��ȡ������
	detector.detect(obj, obj_keypoints);//�õ�ǰ֡������
	detector.detect(scene, scene_keypoints);//�õ���ǰ֡������
	detector.compute(obj, obj_keypoints, obj_descriptors);//����������������������
	detector.compute(scene, scene_keypoints, scene_descriptors);
	BFMatcher matcher(NORM_HAMMING, true); //����������Ϊ���ƶȶ���
	vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches);
	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img); //Դͼ��1��������ƥ��Դͼ��2��������[matches[i]]
	imshow("�˳���ƥ��ǰ", match_img);

	//����ƥ������
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;//��ƥ���Ӧ�Ĳ�ѯͼ����������������������Ծݴ��ҵ�ƥ��㣨ǰ֡�����㣩�ڲ�ѯͼ���е�λ��
		trainIdxs[i] = matches[i].trainIdx;//��ƥ���Ӧ��ѵ��(ģ��)ͼ����������������������Ծݴ��ҵ�ƥ��㣨ǰ֡�����㣩��ѵ��ͼ���е�λ��
	}

	Mat H12;   //�任����

	vector<Point2f> points1; KeyPoint::convert(obj_keypoints, points1, queryIdxs);//ת�Ƶ�points��
	vector<Point2f> points2; KeyPoint::convert(scene_keypoints, points2, trainIdxs);
	int ransacReprojThreshold = 5;  //�ܾ���ֵ


	H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);//ͨ����֡��������Ѱ��͸�ӱ任���� ransacReprojThreshold��ͨ������ransac��³���Է���
	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);//
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //���桮�ڵ㡯
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //���ڵ������
		{
			matchesMask[i1] = 1;//������Щ�㱻����
		}
	}
	Mat match_img2;   //�˳�����㡯��
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);// Scalar::all(-1)����ɫ��� matchesMask������Щ�㱻����

																																  //����Ŀ��λ��
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

	imshow("�˳���ƥ���", match_img2);
	waitKey(0);
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}

// ��HSV�е�Hue�������и���
IplImage *image = 0, *hsv = 0, *hue = 0, *mask = 0, *backproject = 0, *histimg = 0;
CvHistogram *hist = 0;          //ֱ��ͼ��
bool gotBB = false;//Ŀ��ѡȡ��־λ
int backproject_mode = 0;
int select_object = 0;
int track_object = 0;
int show_hist = 1;
CvPoint origin;
CvRect selection;
CvRect track_window;
CvBox2D track_box;              //Meanshift�����㷨���ص�Box��
CvConnectedComp track_comp;
int hdims = 16;                 //����ֱ��ͼbins�ĸ�����Խ��Խ��ȷ
float hranges_arr[] = { 0,180 };  //����ֵ�ķ�Χ
float* hranges = hranges_arr;   //���ڳ�ʼ��CvHistogram��
int vmin = 10, vmax = 65, smin = 30;

void on_mouse(int event, int x, int y, int flags, void* param) //�ú�������ѡ�����Ŀ��
{
	if (!image)
		return;

	if (image->origin)
		y = image->height - y;

	if (select_object) //�������ѡ���������׶Σ����selection�õ�ǰ�����λ�ý�������
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
	case CV_EVENT_LBUTTONDOWN:        //��ʼ���ѡ���������
		origin = cvPoint(x, y);
		selection = cvRect(x, y, 0, 0);  //����
		select_object = 1;            //������ʼ����ѡȡ
		break;
	case CV_EVENT_LBUTTONUP:
		select_object = 0;            //ѡȡ���
		if (selection.width > 0 && selection.height > 0)
			track_object = -1;        //���ѡ��������Ч����򿪸��ٹ���
		gotBB = true;
		break;
	}
}

CvScalar hsv2rgb(float hue)         //���ڽ�Hue��ת����RGB��
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

	return cvScalar(rgb[2], rgb[1], rgb[0], 0);  //���ض�Ӧ����ɫֵ
}

void Cͼ�������Dlg::OnBnClickedButton17()
{
	CvCapture* capture = 0;

	capture = cvCaptureFromAVI("plane.avi"); //��AVI�ļ�

	if (!capture)  //����Ƶ��ʧ�ܴ���
	{
		fprintf(stderr, "Could not initialize capturing...\n");
		return;
	}
	//��ӡ���������б�
	printf("Hot keys: \n"
		"\tESC - quit the program\n"
		"\tc - stop the tracking\n"
		"\tb - switch to/from backprojection view\n"
		"\th - show/hide object histogram\n"
		"To initialize tracking, select the object with mouse\n");

	cvNamedWindow("Histogram", 1);
	cvNamedWindow("CamShiftDemo", 1);
	cvSetMouseCallback("CamShiftDemo", on_mouse, 0);        // �������ص�����
	cvCreateTrackbar("Vmin", "CamShiftDemo", &vmin, 256, 0);//����������
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

	if (!image)  //�տ�ʼ�Ƚ���һЩ������
	{
		/* allocate all the buffers */
		image = cvCreateImage(cvGetSize(frame), 8, 3);
		image->origin = frame->origin;
		hsv = cvCreateImage(cvGetSize(frame), 8, 3);
		hue = cvCreateImage(cvGetSize(frame), 8, 1);
		mask = cvCreateImage(cvGetSize(frame), 8, 1);       //������Ĥͼ��ռ�
		backproject = cvCreateImage(cvGetSize(frame), 8, 1);//���䷴��ͶӰͼ�ռ䣬��Сһ������ͨ��
		hist = cvCreateHist(1, &hdims, CV_HIST_ARRAY, &hranges, 1);  //���佨��ֱ��ͼ�ռ�
		histimg = cvCreateImage(cvSize(320, 200), 8, 3);              //�������ڻ�ֱ��ͼ�Ŀռ�
		cvZero(histimg);   //����Ϊ��ɫ
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


	for (;;)           //������Ƶ֡������ѭ��
	{
		int i, bin_w, c;

		frame = cvQueryFrame(capture);
		if (!frame)
			break;


		cvCopy(frame, image, 0);
		cvCvtColor(image, hsv, CV_BGR2HSV);   // ��ͼ���RGB��ɫϵתΪHSV��ɫϵ

		if (track_object)  //   �����ǰ����Ҫ���ٵ�����   
		{
			int _vmin = vmin, _vmax = vmax;

			//��Ĥ�壬ֻ��������ֵΪH��0~180��S��smin~256��V��vmin~vmax֮��Ĳ���
			void cvInRangeS(const CvArr* src, CvScalar lower, CvScalar upper, CvArr* dst);
			cvInRangeS(hsv, cvScalar(0, smin, MIN(_vmin, _vmax), 0),
				cvScalar(180, 256, MAX(_vmin, _vmax), 0), mask);
			cvSplit(hsv, hue, 0, 0, 0); // ȡ��H����

			if (track_object < 0) //�����Ҫ���ٵ����廹û�н���������ȡ�������ѡȡ�����ͼ��������ȡ
			{
				float max_val = 0.f;
				cvSetImageROI(hue, selection);  // ����ԭѡ���
				cvSetImageROI(mask, selection); // ����Mask��ѡ���
				cvCalcHist(&hue, hist, 0, mask);// �õ�ѡ�������������Ĥ���ڵ�ֱ��ͼ
				cvGetMinMaxHistValue(hist, 0, &max_val, 0, 0);
				cvConvertScale(hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0);  // ��ֱ��ͼתΪ0~255
				cvResetImageROI(hue);  // remove ROI
				cvResetImageROI(mask);
				track_window = selection;
				track_object = 1;

				cvZero(histimg);
				bin_w = histimg->width / hdims;
				for (i = 0; i < hdims; i++)  //��ֱ��ͼ��ͼ��ռ�
				{
					int val = cvRound(cvGetReal1D(hist->bins, i)*histimg->height / 255);
					//��һ��double�͵��������������룬������һ��������
					CvScalar color = hsv2rgb(i*180.f / hdims);
					cvRectangle(histimg, cvPoint(i*bin_w, histimg->height),
						cvPoint((i + 1)*bin_w, histimg->height - val),
						color, -1, 8, 0);
				}
			}

			// �õ�hue�ķ���ͶӰͼ           
			cvCalcBackProject(&hue, backproject, hist);
			// �õ�����ͶӰͼmask�ڵ�����
			cvAnd(backproject, mask, backproject, 0);
			cvCamShift(backproject, track_window,
				cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1),
				&track_comp, &track_box);
			//ʹ��MeanShift�㷨��backproject�е����ݽ������������ظ��ٽ��

			track_window = track_comp.rect; //�õ����ٽ���ľ��ο�

			if (backproject_mode)
				cvCvtColor(backproject, image, CV_GRAY2BGR); // ��ʾģʽ
			if (!image->origin)
				track_box.angle = -track_box.angle;
			cvEllipseBox(image, track_box, CV_RGB(255, 0, 0), 3, CV_AA, 0); //�������ٽ����λ��
		}

		if (select_object && selection.width > 0 && selection.height > 0) //�������������ѡ�񣬻���ѡ���
		{
			cvSetImageROI(image, selection);
			cvXorS(image, cvScalarAll(255), image, 0);
			cvResetImageROI(image);
		}

		cvShowImage("CamShiftDemo", image); //��ʾ��Ƶ��ֱ��ͼ
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


	// TODO: �ڴ���ӿؼ�֪ͨ����������
}
