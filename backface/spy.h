#pragma once
#include "utils.h"
#include <windows.h>
#include <tchar.h>
#include <vector>
#include <deque>
#include <functional>
#include <thread>
#include <engine/gtp.h>

 
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
    std::shared_ptr<IGtpAgent> agent;
};


constexpr int max_stone_templates = 3;

struct TemplateResource {

	int stone_size;
	std::vector<BYTE> stoneImages[max_stone_templates];
	std::vector<BYTE> stoneMask;
	std::vector<BYTE> lastMoveMask;
	std::vector<BYTE> blackImage;
	std::vector<BYTE> whiteImage;
};


class BoardSpy : public BoardSinker {
public:

	BoardSpy();
	~BoardSpy();

	// callbacks
	std::function<void()> onSizeChanged;

	POINT coord2Screen(int x, int y) const;
	bool found() const;

	void initResource();

	void placeAt(int x, int y);

	bool bindWindow(HWND hWnd);

	bool routineCheck();
	
protected:
	bool initBitmaps();
	bool scanBoard(int data[], int& lastMove); 
	int detectStone(int move, bool& isLastMove) const;
	double compareBoardRegionAt(int move, const std::vector<BYTE>& stone, const std::vector<BYTE>& mask) const;

	bool locateStartPosition(Hdib&, int& startx, int& starty);
	bool calcBoardPositions(HWND hwnd, int startx, int starty);

protected:
	bool initialBoard();
	void releaseWindows();





private:

	int offsetX_;
	int offsetY_;
	int stoneSize_;

	int tplStoneSize_;
	int tplBoardSize_;

	std::vector<BYTE> stoneImages_[max_stone_templates];
	std::vector<BYTE> blackStoneData_;
	std::vector<BYTE> whiteStoneData_;
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

	Hdib boardDIB_;

	HBitmap hBoardBitmap_;
	Hdc hBoardDC_;

	HWND hTargetWnd;
	RECT lastRect_;
	int error_count_;
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
