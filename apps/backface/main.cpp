#include "spy.h"
#include <Windows.h>
#include <Windowsx.h>
#include <CommCtrl.h>
#include "resource.h"
#include <map>


#pragma comment(linker, \
  "\"/manifestdependency:type='Win32' "\
  "name='Microsoft.Windows.Common-Controls' "\
  "version='6.0.0.0' "\
  "processorArchitecture='*' "\
  "publicKeyToken='6595b64144ccf1df' "\
  "language='*'\"")

#pragma comment    (lib, "comctl32.lib")   

#define WINDOW_FINDER_APP_MUTEX_NAME	_T("WINDOWFINDERMUTEX")
#define BULLSEYE_CENTER_X_OFFSET		15
#define BULLSEYE_CENTER_Y_OFFSET		18




long RefreshWindow(HWND hwndWindowToBeRefreshed);
long HighlightFoundWindow(HWND hwndFoundWindow);



BOOL InitializeApplication
(
	HINSTANCE hThisInst,
	HINSTANCE hPrevInst,
	LPTSTR lpszArgs,
	int nWinMode
	);
BOOL UninitializeApplication();
BOOL InitialiseResources();
BOOL UninitialiseResources();



HWND		g_hwndFoundWindow = NULL;
HPEN		g_hRectanglePen = NULL;
HCURSOR		g_hCursorPrevious = NULL;
HBITMAP		g_hBitmapFinderToolFilled;
HBITMAP		g_hBitmapFinderToolEmpty;
HCURSOR		g_hCursorSearchWindow = NULL;
HINSTANCE	g_hInst = NULL;
HANDLE		g_hApplicationMutex = NULL;
DWORD		g_dwLastError = 0;

TCHAR		g_szLocalPath[256];


Dialog* Dialog::current = nullptr;


Dialog* Dialog::create(HINSTANCE hInst) {
	auto hDlg = CreateDialogParam(hInst, MAKEINTRESOURCE(IDD_DIALOG_MAIN_WINDOW), 0, Proc, 0);
	//current is assigned in INIT_DIALOG
	return current;
}


void Dialog::show(int nCmdShow) {
	ShowWindow(hWnd_, nCmdShow);
}

int Dialog::messageLoop() {

	BOOL ret;
	MSG msg;

	while ((ret = GetMessage(&msg, 0, 0, 0)) != 0) {
		if (ret == -1) /* error found */
			return -1;

		if (!IsDialogMessage(hWnd_, &msg)) {
			TranslateMessage(&msg); /* translate virtual-key messages */
			DispatchMessage(&msg); /* send it to dialog procedure */
		}
	}
	return 0;
}

int WINAPI _tWinMain(HINSTANCE hInst, HINSTANCE h0, LPTSTR lpCmdLine, int nCmdShow) {

	g_hInst = hInst;

	InitializeApplication(hInst, h0, lpCmdLine, nCmdShow);
	InitialiseResources();

	InitCommonControls();

	auto* dlg = Dialog::create(hInst);
	dlg->show(nCmdShow);
	if (dlg->messageLoop() < 0)
		return -1;

	UninitializeApplication();
	UninitialiseResources();
	delete dlg;
	return 0;
}





#define IDT_TIMER1 100
#define IDT_TIMER_LABEL 101

Dialog::Dialog(HWND wnd)
: hWnd_(wnd)
, wndLabel_(spy)
{
	bStartSearchWindow = FALSE;
}

Dialog::~Dialog()
{
	current = nullptr;
}

INT_PTR CALLBACK Dialog::Proc(HWND hDlg, UINT uMsg, WPARAM wParam, LPARAM lParam) {

	if (!current) {
		if (uMsg == WM_INITDIALOG) {
			current = new Dialog(hDlg);
			current->onInit();
		}

		return FALSE;
	}

	return current->Proc(uMsg, wParam, lParam);
}


INT_PTR CALLBACK Dialog::Proc(UINT uMsg, WPARAM wParam, LPARAM lParam) {

	switch (uMsg) {
	case WM_CLOSE:
		onClose();
		return TRUE;
	case WM_DESTROY:
		onDestroy();
		return TRUE;
	case WM_TIMER:
		switch (wParam) {
		case IDT_TIMER1:
			onDetectInterval();
			return TRUE;
		}
		return FALSE;
	case WM_COMMAND: {
		WORD wID = LOWORD(wParam);         // item, control, or accelerator identifier 

		if (wID == IDC_STATIC_ICON_FINDER_TOOL)
		{
			// Because the IDC_STATIC_ICON_FINDER_TOOL static control is set with the SS_NOTIFY
			// flag, the Search Window's dialog box will be sent a WM_COMMAND message when this 
			// static control is clicked.
			// We start the window search operation by calling the DoSearchWindow() function.
			SearchWindow();
			return TRUE;
		}

		if (wID == IDC_BTN_PONDER) {
			spy.stopThink();
			return TRUE;
		}

		if (wID == IDC_CHECK_AUTO) {
			bool hint = IsDlgButtonChecked(hWnd_, wID);
			auto_play_ = hint;
			return TRUE;
		}
	}

	case WM_MOUSEMOVE:
		if (bStartSearchWindow)
		{
			// Only when we have started the Window Searching operation will we 
			// track mouse movement.
			DoMouseMove(uMsg, wParam, lParam);
		}

		return TRUE;

	case WM_LBUTTONUP:
	{
		if (bStartSearchWindow)
		{
			// Only when we have started the window searching operation will we
			// be interested when the user lifts up the left mouse button.
			DoMouseUp(uMsg, wParam, lParam);
		}

		return TRUE;
	}
	}
	return 0;
}

void Dialog::onInit() {

	auto_play_ = false;
	connected_ = false;

	spy.initResource();

	SetTimer(hWnd_,             // handle to main window 
		IDT_TIMER1,            // timer identifier 
		10,                 // 10-second interval 
		(TIMERPROC)NULL);     // no timer callback 

	wndLabel_.create(hWnd_);

	spy.onMoveChange = [&](char player, int x, int y, int steps) {

		wndLabel_.hide();

		TCHAR wbuf[50];
		wsprintf(wbuf, _T("%d: %s (%d,%d)"), steps, player == 1 ? _T("B") : _T("W"), x, y);
		if (steps == 1)
			lstrcpy(szMoves_, _T("[                                         ]\r\n"));

		lstrcat(szMoves_, wbuf);
		lstrcat(szMoves_, _T("\r\n"));
		memcpy(&szMoves_[1], wbuf, lstrlen(wbuf)*sizeof(TCHAR));
		SetDlgItemText(hWnd_, IDC_EDIT_STATUS, szMoves_);
	};

	spy.onSizeChanged = [&]() {
		wndLabel_.update();
	};

	spy.onMovePredict = [&](int player, int x, int y) {

		bool autoplay = auto_play_;

		int label = y*19 + x;
		wndLabel_.setPos(label);
		wndLabel_.show();
		if (autoplay) {
			spy.placeAt(x, y);
		}	
	};

	spy.onMovePass = [&]() {
		MessageBox(NULL, _T("PASS"), TEXT("spy"), MB_ICONINFORMATION);
	};

	spy.onResign = [&]() {
		MessageBox(NULL, _T("RESIGN"), TEXT("spy"), MB_ICONINFORMATION);
	};

	spy.onGtpReady = [&]() {
		updateStatus();
	};
}

void Dialog::updateStatus() {
	TCHAR msg[128];

	if (connected_) {
		lstrcpy(msg, _T("Located, "));
	}
	else {
		lstrcpy(msg, _T("Not Found, "));
	}

	SetDlgItemText(hWnd_, IDC_STATIC_INFO, msg);
}

void Dialog::onDetectInterval() {

	bool r = spy.detecteBoard();

	if (r != connected_) {
		connected_ = r;
		updateStatus();
	}
}

void Dialog::onClose() {
	//DestroyWindow(hWnd_);
	if (MessageBox(hWnd_, TEXT("Exit?"), TEXT("Exit"),
		MB_ICONQUESTION | MB_YESNO) == IDYES)
	{
		DestroyWindow(hWnd_);
	}
}

void Dialog::onDestroy() {
	PostQuitMessage(0);
}



// Synopsis :
// 1. This routine moves the mouse cursor hotspot to the exact 
// centre position of the bullseye in the finder tool static control.
//
// 2. This function, when used together with DoSetFinderToolImage(),
// gives the illusion that the bullseye image has indeed been transformed
// into a cursor and can be moved away from the Finder Tool Static
// control.
BOOL MoveCursorPositionToBullsEye(HWND hwndDialog)
{
	BOOL bRet = FALSE;
	HWND hwndToolFinder = NULL;
	RECT rect;
	POINT screenpoint;

	// Get the window handle of the Finder Tool static control.
	hwndToolFinder = GetDlgItem(hwndDialog, IDC_STATIC_ICON_FINDER_TOOL);

	if (hwndToolFinder)
	{
		// Get the screen coordinates of the static control,
		// add the appropriate pixel offsets to the center of 
		// the bullseye and move the mouse cursor to this exact
		// position.
		GetWindowRect(hwndToolFinder, &rect);
		screenpoint.x = rect.left + BULLSEYE_CENTER_X_OFFSET;
		screenpoint.y = rect.top + BULLSEYE_CENTER_Y_OFFSET;
		SetCursorPos(screenpoint.x, screenpoint.y);
	}

	return bRet;
}

// Synopsis :
// 1. This function starts the window searching operation.
//
// 2. A very important part of this function is to capture 
// all mouse activities from now onwards and direct all mouse 
// messages to the "Search Window" dialog box procedure.
long Dialog::SearchWindow() {

	bStartSearchWindow = TRUE;

	// Display the empty window bitmap image in the Finder Tool static control.
	SetFinderToolImage(FALSE);

	MoveCursorPositionToBullsEye(hWnd_);

	// Set the screen cursor to the BullsEye cursor.
	if (g_hCursorSearchWindow)
	{
		g_hCursorPrevious = SetCursor(g_hCursorSearchWindow);
	}
	else
	{
		g_hCursorPrevious = NULL;
	}

	// Very important : capture all mouse activities from now onwards and
	// direct all mouse messages to the "Search Window" dialog box procedure.
	SetCapture(hWnd_);

	return 0;
}

// Synopsis :
// 1. This routine sets the Finder Tool icon to contain an appropriate bitmap.
//
// 2. If bSet is TRUE, we display the BullsEye bitmap. Otherwise the empty window
// bitmap is displayed.
BOOL Dialog::SetFinderToolImage(BOOL bSet)
{
	HBITMAP hBmpToSet = NULL;
	BOOL bRet = TRUE;

	if (bSet)
	{
		// Set a FILLED image.
		hBmpToSet = g_hBitmapFinderToolFilled;
	}
	else
	{
		// Set an EMPTY image.
		hBmpToSet = g_hBitmapFinderToolEmpty;
	}

	SendDlgItemMessage
		(
			(HWND)hWnd_, // handle of dialog box 
			(int)IDC_STATIC_ICON_FINDER_TOOL, // identifier of control 
			(UINT)STM_SETIMAGE, // message to send 
			(WPARAM)IMAGE_BITMAP, // first message parameter 
			(LPARAM)hBmpToSet // second message parameter 
			);

	return bRet;
}



BOOL InitialiseResources()
{
	BOOL bRet = FALSE;

	g_hCursorSearchWindow = LoadCursor(g_hInst, MAKEINTRESOURCE(IDC_CURSOR_SEARCH_WINDOW));
	if (g_hCursorSearchWindow == NULL)
	{
		bRet = FALSE;
		goto InitialiseResources_0;
	}

	g_hRectanglePen = CreatePen(PS_SOLID, 3, RGB(255, 0, 0));
	if (g_hRectanglePen == NULL)
	{
		bRet = FALSE;
		goto InitialiseResources_0;
	}

	g_hBitmapFinderToolFilled = LoadBitmap(g_hInst, MAKEINTRESOURCE(IDB_BITMAP_FINDER_FILLED));
	if (g_hBitmapFinderToolFilled == NULL)
	{
		bRet = FALSE;
		goto InitialiseResources_0;
	}

	g_hBitmapFinderToolEmpty = LoadBitmap(g_hInst, MAKEINTRESOURCE(IDB_BITMAP_FINDER_EMPTY));
	if (g_hBitmapFinderToolEmpty == NULL)
	{
		bRet = FALSE;
		goto InitialiseResources_0;
	}

	// All went well. Return TRUE.
	bRet = TRUE;

InitialiseResources_0:

	return bRet;
}

BOOL InitializeApplication
(
	HINSTANCE hThisInst,
	HINSTANCE hPrevInst,
	LPTSTR lpszArgs,
	int nWinMode
	)
{
	BOOL bRetTemp = FALSE;
	long lRetTemp = 0;
	BOOL bRet = FALSE;
	TCHAR* pos;

	// Create the application mutex so that no two instances of this app can run.
	g_hApplicationMutex = CreateMutex
		(
			(LPSECURITY_ATTRIBUTES)NULL, // pointer to security attributes 
			(BOOL)TRUE, // flag for initial ownership 
			(LPCTSTR)WINDOW_FINDER_APP_MUTEX_NAME // pointer to mutex-object name 
			);

	g_dwLastError = GetLastError();

	// If cannot create Mutex, exit.
	if (g_hApplicationMutex == NULL)
	{
		bRet = FALSE;
		goto InitializeApplication_0;
	}

	// If Mutex already existed, exit.
	if (g_dwLastError == ERROR_ALREADY_EXISTS)
	{
		CloseHandle(g_hApplicationMutex);
		g_hApplicationMutex = NULL;
		bRet = FALSE;
		goto InitializeApplication_0;
	}

	GetModuleFileName(hThisInst, g_szLocalPath, sizeof(g_szLocalPath) / sizeof(TCHAR) - 1);
	pos = _tcsrchr(g_szLocalPath, _T('\\'));
	*pos = 0;

	// All went well, return a TRUE.
	bRet = TRUE;

InitializeApplication_0:

	return bRet;
}


// Synopsis :
// 1. This is the handler for WM_MOUSEMOVE messages sent to the "Search Window" dialog proc.
//
// 2. Note that we do not handle every WM_MOUSEMOVE message sent. Instead, we check to see 
// if "g_bStartSearchWindow" is TRUE. This BOOL will be set to TRUE when the Window
// Searching Operation is actually started. See the WM_COMMAND message handler in 
// SearchWindowDialogProc() for more details.
//
// 3. Because the "Search Window" dialog immediately captures the mouse when the Search Operation 
// is started, all mouse movement is monitored by the "Search Window" dialog box. This is 
// regardless of whether the mouse is within or without the "Search Window" dialog. 
//
// 4. One important note is that the horizontal and vertical positions of the mouse cannot be 
// calculated from "lParam". These values can be inaccurate when the mouse is outside the
// dialog box. Instead, use the GetCursorPos() API to capture the position of the mouse.
long Dialog::DoMouseMove
(
	UINT message,
	WPARAM wParam,
	LPARAM lParam
	)
{
	POINT		screenpoint;
	TCHAR		szText[256];
	long		lRet = 0;

	// Must use GetCursorPos() instead of calculating from "lParam".
	GetCursorPos(&screenpoint);

	// Display global positioning in the dialog box.
	wsprintf(szText, _T("(%d, %d)"), screenpoint.x, screenpoint.y);
	SetDlgItemText(hWnd_, IDC_STATIC_POS, szText);

	// Determine the window that lies underneath the mouse cursor.
	auto hWndToCheck = WindowFromPoint(screenpoint);

	// Check first for validity.
	if (CheckWindowValidity(hWndToCheck))
	{
		// We have just found a new window.

		// Display some information on this found window.
		DisplayInfoOnFoundWindow(hWndToCheck);

		// If there was a previously found window, we must instruct it to refresh itself. 
		// This is done to remove any highlighting effects drawn by us.
		if (g_hwndFoundWindow)
		{
			RefreshWindow(g_hwndFoundWindow);
		}

		// Indicate that this found window is now the current global found window.
		g_hwndFoundWindow = hWndToCheck;

		// We now highlight the found window.
		HighlightFoundWindow(g_hwndFoundWindow);
	}

	return lRet;
}



// Synopsis :
// 1. Handler for WM_LBUTTONUP message sent to the "Search Window" dialog box.
// 
// 2. We restore the screen cursor to the previous one.
//
// 3. We stop the window search operation and release the mouse capture.
long Dialog::DoMouseUp
(
	UINT message,
	WPARAM wParam,
	LPARAM lParam
	)
{
	// If we had a previous cursor, set the screen cursor to the previous one.
	// The cursor is to stay exactly where it is currently located when the 
	// left mouse button is lifted.
	if (g_hCursorPrevious)
	{
		SetCursor(g_hCursorPrevious);
	}

	// If there was a found window, refresh it so that its highlighting is erased. 
	if (g_hwndFoundWindow)
	{
		RefreshWindow(g_hwndFoundWindow);
	}

	// Set the bitmap on the Finder Tool icon to be the bitmap with the bullseye bitmap.
	SetFinderToolImage(TRUE);

	// Very important : must release the mouse capture.
	ReleaseCapture();

	// Set the global search window flag to FALSE.
	bStartSearchWindow = FALSE;

	if (IsWindow(g_hwndFoundWindow)) {
		spy.setWindow(g_hwndFoundWindow);
	}

	return 0;
}



// Synopsis :
// 1. This function checks a hwnd to see if it is actually the "Search Window" Dialog's or Main Window's
// own window or one of their children. If so a FALSE will be returned so that these windows will not
// be selected. 
//
// 2. Also, this routine checks to see if the hwnd to be checked is already a currently found window.
// If so, a FALSE will also be returned to avoid repetitions.
BOOL Dialog::CheckWindowValidity(HWND hwndToCheck)
{
	HWND hwndTemp = NULL;
	BOOL bRet = TRUE;

	// The window must not be NULL.
	if (hwndToCheck == NULL)
	{
		bRet = FALSE;
		goto CheckWindowValidity_0;
	}

	// It must also be a valid window as far as the OS is concerned.
	if (IsWindow(hwndToCheck) == FALSE)
	{
		bRet = FALSE;
		goto CheckWindowValidity_0;
	}

	// Ensure that the window is not the current one which has already been found.
	if (hwndToCheck == g_hwndFoundWindow)
	{
		bRet = FALSE;
		goto CheckWindowValidity_0;
	}

	// It also must not be the "Search Window" dialog box itself.
	if (hwndToCheck == hWnd_)
	{
		bRet = FALSE;
		goto CheckWindowValidity_0;
	}

CheckWindowValidity_0:

	return bRet;
}


long Dialog::DisplayInfoOnFoundWindow(HWND hwndFoundWindow)
{
	RECT		rect;              // Rectangle area of the found window.
	TCHAR		szText[256];
	TCHAR		szClassName[100];

	// Get the screen coordinates of the rectangle of the found window.
	GetWindowRect(hwndFoundWindow, &rect);

	// Get the class name of the found window.
	GetClassName(hwndFoundWindow, szClassName, sizeof(szClassName)/sizeof(TCHAR) - 1);


	// Display some information on the found window.
	wsprintf
		(
			szText, _T("Handle: 0x%08X\r\nClass : %s\r\nleft: %d\r\ntop: %d\r\nright: %d\r\nbottom: %d\r\n"),
			hwndFoundWindow,
			szClassName,
			rect.left,
			rect.top,
			rect.right,
			rect.bottom
			);

	SetDlgItemText(hWnd_, IDC_EDIT_STATUS, szText);

	return 0;
}


long RefreshWindow(HWND hwndWindowToBeRefreshed)
{
	long lRet = 0;

	InvalidateRect(hwndWindowToBeRefreshed, NULL, TRUE);
	UpdateWindow(hwndWindowToBeRefreshed);
	RedrawWindow(hwndWindowToBeRefreshed, NULL, NULL, RDW_FRAME | RDW_INVALIDATE | RDW_UPDATENOW | RDW_ALLCHILDREN);

	return lRet;
}



BOOL UninitializeApplication()
{
	BOOL bRet = TRUE;

	if (g_hApplicationMutex)
	{
		ReleaseMutex(g_hApplicationMutex);
		CloseHandle(g_hApplicationMutex);
		g_hApplicationMutex = NULL;
	}

	return bRet;
}

BOOL UninitialiseResources()
{
	BOOL bRet = TRUE;

	if (g_hCursorSearchWindow)
	{
		// No need to destroy g_hCursorSearchWindow. It was not created using 
		// CreateCursor().
	}

	if (g_hRectanglePen)
	{
		bRet = DeleteObject(g_hRectanglePen);
		g_hRectanglePen = NULL;
	}

	if (g_hBitmapFinderToolFilled)
	{
		DeleteObject(g_hBitmapFinderToolFilled);
		g_hBitmapFinderToolFilled = NULL;
	}

	if (g_hBitmapFinderToolEmpty)
	{
		DeleteObject(g_hBitmapFinderToolEmpty);
		g_hBitmapFinderToolEmpty = NULL;
	}

	return bRet;
}


long HighlightFoundWindow(HWND hwndFoundWindow) {
	HDC		hWindowDC = NULL;  // The DC of the found window.
	HGDIOBJ	hPrevPen = NULL;   // Handle of the existing pen in the DC of the found window.
	HGDIOBJ	hPrevBrush = NULL; // Handle of the existing brush in the DC of the found window.
	RECT		rect;              // Rectangle area of the found window.
	long		lRet = 0;

	// Get the screen coordinates of the rectangle of the found window.
	GetWindowRect(hwndFoundWindow, &rect);

	// Get the window DC of the found window.
	hWindowDC = GetWindowDC(hwndFoundWindow);

	if (hWindowDC)
	{
		// Select our created pen into the DC and backup the previous pen.
		hPrevPen = SelectObject(hWindowDC, g_hRectanglePen);

		// Select a transparent brush into the DC and backup the previous brush.
		hPrevBrush = SelectObject(hWindowDC, GetStockObject(HOLLOW_BRUSH));

		// Draw a rectangle in the DC covering the entire window area of the found window.
		Rectangle(hWindowDC, 0, 0, rect.right - rect.left, rect.bottom - rect.top);

		// Reinsert the previous pen and brush into the found window's DC.
		SelectObject(hWindowDC, hPrevPen);

		SelectObject(hWindowDC, hPrevBrush);

		// Finally release the DC.
		ReleaseDC(hwndFoundWindow, hWindowDC);
	}

	return lRet;
}


PredictWnd* PredictWnd::current = nullptr;

void PredictWnd::create(HWND hParent) {
	current = this;
	hWnd = CreateDialog(g_hInst, MAKEINTRESOURCE(IDD_DIALOG1), hParent, Proc);
}

INT_PTR CALLBACK PredictWnd::Proc(HWND hDlg, UINT uMsg, WPARAM wParam, LPARAM lParam) {

	if (uMsg == WM_INITDIALOG) {
		current->hWnd = hDlg;
		current->onInit();
		return FALSE;
	}

	if (!current)
		return FALSE;

	return current->Proc(uMsg, wParam, lParam);
}

INT_PTR CALLBACK PredictWnd::Proc(UINT uMsg, WPARAM wParam, LPARAM lParam) {

	switch (uMsg) {
	case WM_ERASEBKGND:
	{
		HBRUSH brush;
		RECT rect;
		brush = CreateSolidBrush(RGB(255, 0, 0));
		SelectObject((HDC)wParam, brush);
		GetClientRect(hWnd, &rect);
		Rectangle((HDC)wParam, rect.left, rect.top, rect.right, rect.bottom);
		return TRUE;
	}

	case WM_MOUSEMOVE: {
		mouseOver = true;
		ShowWindow(hWnd, SW_HIDE);

		SetTimer(hWnd,             // handle to main window 
			IDT_TIMER1,            // timer identifier 
			500,                 // 10-second interval 
			(TIMERPROC)NULL);     // no timer callback 

		return TRUE;
	}
	case WM_TIMER: {
		switch (wParam) {
		case IDT_TIMER1:
			mouseOver = false;
			if (!hiding_)
				ShowWindow(hWnd, SW_SHOW);
			return TRUE;
		}
		return TRUE;
	}
	}
	return FALSE;
}

void PredictWnd::onInit() {
	HRGN hrgnShape = createRegion();
	SetWindowPos(hWnd, NULL, 0, 0, size, size, 0);
	SetWindowRgn(hWnd, hrgnShape, TRUE);
}

PredictWnd::PredictWnd(BoardSpy& spy)
: hWnd(NULL)
, spy_(spy)
, mouseOver(false){
	if (!loadRes())
		fatal(_T("cannot find label resource"));
}


bool PredictWnd::loadRes() {
	TCHAR path[256];

	lstrcpy(path, g_szLocalPath);
	lstrcat(path, _T("\\res\\label.bmp"));

	std::vector<BYTE> data;
	int width, height;
	if (!loadBmpData(path, data, width, height))
		return false;

	if (width != height)
		return false;

	size = width;

	label_mask_data.resize(width*height);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++) {
			auto idx = i*width + j;
			auto target = i*width * 3 + j * 3;
			auto r = data[target];
			auto g = data[target + 1];
			auto b = data[target + 2];

			if (r != 0 || g != 0 || b != 255) {
				label_mask_data[idx] = 1;
			}
			else {
				label_mask_data[idx] = 0;
			}
		}
	}
	return true;
}


HRGN PredictWnd::createRegion() {

	// We start with the full rectangular region
	HRGN hrgnBitmap = ::CreateRectRgn(0, 0, size, size);

	BOOL bInTransparency = FALSE;  // Already inside a transparent part?
	int start_x = -1;			   // Start of transparent part

								   // For all rows of the bitmap ...
	for (int y = 0; y < size; y++)
	{
		// For all pixels of the current row ...
		// (To close any transparent region, we go one pixel beyond the
		// right boundary)
		for (int x = 0; x <= size; x++)
		{
			BOOL bTransparent = FALSE; // Current pixel transparent?

									   // Check for positive transparency within image boundaries
			if ((x < size) && (label_mask_data[y*size + x] == 0))
			{
				bTransparent = TRUE;
			}

			// Does transparency change?
			if (bInTransparency != bTransparent)
			{
				if (bTransparent)
				{
					// Transparency starts. Remember x position.
					bInTransparency = TRUE;
					start_x = x;
				}
				else
				{
					// Transparency ends (at least beyond image boundaries).
					// Create a region for the transparency, one pixel high,
					// beginning at start_x and ending at the current x, and
					// subtract that region from the current bitmap region.
					// The remainding region becomes the current bitmap region.
					HRGN hrgnTransp = ::CreateRectRgn(start_x, y, x, y + 1);
					::CombineRgn(hrgnBitmap, hrgnBitmap, hrgnTransp, RGN_DIFF);
					::DeleteObject(hrgnTransp);

					bInTransparency = FALSE;
				}
			}
		}
	}

	return hrgnBitmap;
}

void PredictWnd::setPos(int movePos) {
	move_ = movePos;
	auto x = movePos % 19, y = movePos / 19;
	auto pt = spy_.coord2Screen(x, y);
	SetWindowPos(hWnd, NULL, pt.x - size / 2, pt.y - size / 2, size, size, 0);
}

void PredictWnd::update() {
	setPos(move_);
}

void PredictWnd::show() {

	hiding_ = false;

	if (!mouseOver)
		ShowWindow(hWnd, SW_SHOW);
}

void PredictWnd::hide() {
	hiding_ = true;
	ShowWindow(hWnd, SW_HIDE);
}

