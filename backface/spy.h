#pragma once
#include "utils.h"
#include <windows.h>
#include <tchar.h>
#include <vector>
#include <deque>
#include <functional>
#include <thread>
#include <engine/aq/aq.h>
#include <simplemodel/zero_model.hpp>


class BoardSinker {
public:
	std::function<void()> onGtpReady;
	std::function<void()> onMovePass;
	std::function<void()> onResign;
	std::function<void(int player, int x, int y)> onMovePredict;
	std::function<void(int player, int x, int y, int steps)> onMoveChange;

public:
	void stopThink();
	
protected:
	void init(const std::string& cfg_path);
	void deinit();

	void newGame(int mycolor);
	void commitMove(int player, int x, int y);

protected:
	int myTurn_;
	int turn_;
	int steps_;

	std::shared_ptr<Gtp> gtp;
	std::unique_ptr<ZeroPredictModel> model;
    std::unique_ptr<IGtpAgent> agent;
};


class BoardSpy : public BoardSinker {
public:
	static const int max_stone_templates = 3;

	BoardSpy();
	~BoardSpy();

	// callbacks
	std::function<void()> onSizeChanged;

	POINT coord2Screen(int x, int y) const;
	bool found() const;

	void initResource();
	bool detecteBoard();
	void setWindow(HWND hwnd);
	void placeAt(int x, int y);
	
protected:
	bool initBitmaps();
	bool setWindowInternal(HWND hwnd);
	bool scanBoard(int data[], int& lastMove); 
	int detectStone(int move, bool& isLastMove) const;
	double compareBoardRegionAt(int move, const std::vector<BYTE>& stone, const std::vector<BYTE>& mask) const;

protected:
	bool reAttach();
	void releaseWindows();


private:
	bool findBoard();

	int offsetX_;
	int offsetY_;
	int stoneSize_;

	int tplStoneSize_;
	int tplBoardSize_;

	std::vector<BYTE> stoneImages_[max_stone_templates];
	std::vector<BYTE> stoneMaskData_;
	std::vector<BYTE> lastMoveMaskData_;
	std::vector<BYTE> blackImage_;
	std::vector<BYTE> whiteImage_;

private:
	int board[361];
	int board_last[361];
	int board_age[361];

private:
	// Window Resources
	HDC hDisplayDC_;
	int nDispalyBitsCount_;
	int nBpp_;

	BITMAPINFO boardBitmapInfo_;
	std::vector<BYTE> boardDIBs_;

	HBitmap hBoardBitmap_;
	Hdc hBoardDC_;

	HWND hTargetWnd;
	HDC hTargetDC;
	RECT targetRect_;
};



class PredictWnd {
public:
	PredictWnd(BoardSpy& spy);
	void create(HWND hParent);
	void show();
	void setPos(int movePos);
	void update();
	void hide();

protected:
	bool loadRes();
	HRGN createRegion();

	static INT_PTR CALLBACK Proc(HWND hDlg, UINT uMsg, WPARAM wParam, LPARAM lParam);

	INT_PTR CALLBACK Proc(UINT uMsg, WPARAM wParam, LPARAM lParam);

	void onInit();

	std::vector<BYTE> label_mask_data;
	int move_;
	int size;
	HWND hWnd;

	static PredictWnd* current;

private:
	BoardSpy& spy_;

	bool mouseOver;
	bool hiding_;
};


class Dialog {

public:
	static Dialog* create(HINSTANCE hInst);
	
	static INT_PTR CALLBACK Proc(HWND hDlg, UINT uMsg, WPARAM wParam, LPARAM lParam);


	Dialog(HWND);
	~Dialog();
	void show(int nCmdShow);
	int messageLoop();

protected:
	void onInit();
	void onClose();
	void onDestroy();
	void onDetectInterval();

	long DoMouseMove
		(
			UINT message,
			WPARAM wParam,
			LPARAM lParam
			);
	long DoMouseUp
		(
			UINT message,
			WPARAM wParam,
			LPARAM lParam
			);

	long SearchWindow();
	BOOL CheckWindowValidity(HWND hwndToCheck);
	long DisplayInfoOnFoundWindow(HWND hwndFoundWindow);
	BOOL SetFinderToolImage(BOOL bSet);

	void updateStatus();

protected:
	static Dialog* current;

	INT_PTR CALLBACK Proc(UINT uMsg, WPARAM wParam, LPARAM lParam);

	HWND hWnd_;
	BoardSpy spy;

	BOOL bStartSearchWindow;

	PredictWnd wndLabel_;

	TCHAR szMoves_[300 * 50];
	bool connected_;
	bool auto_play_;
};



extern TCHAR		g_szLocalPath[256];


void fatal(LPCTSTR msg);
bool loadBmpData(LPCTSTR pszFile, std::vector<BYTE>& img, int& width, int& height);


void gtp_routine();
