
// ͼ�������.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// Cͼ�������App: 
// �йش����ʵ�֣������ ͼ�������.cpp
//

class Cͼ�������App : public CWinApp
{
public:
	Cͼ�������App();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern Cͼ�������App theApp;