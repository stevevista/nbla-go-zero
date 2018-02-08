
#include "spy.h"

#undef min

//#pragma comment(lib, "nngo.lib")
//#pragma comment(lib, "cudnn.lib")
//#pragma comment(lib, "cudart_static.lib")



const int THRESH_STONE_EXISTS_INTERVAL = 50;


static double colorDiff(BYTE r, BYTE g, BYTE b, BYTE r2, BYTE g2, BYTE b2) {
	return sqrt((r - r2)*(r - r2) + (g - g2)*(g - g2) + (b - b2)*(b - b2));
}


struct stone_template {
	int value;
	LPCTSTR file;
};


static stone_template stone_files[BoardSpy::max_stone_templates] = {
	{ 1, _T("black.bmp") },
	{ -1, _T("white.bmp") },
	{ 0, _T("empty.bmp") },
};

static LPCTSTR mask_files[] = {
	_T("stone.bmp"),
	_T("d.bmp"),
};

BoardSpy::BoardSpy() 
: hTargetWnd(NULL)
, hTargetDC(NULL)
{
	memset(board, 0, sizeof(board));
}

BoardSpy::~BoardSpy() {

	deinit();
	releaseWindows();

	if (hDisplayDC_)
		ReleaseDC(NULL, hDisplayDC_);
}


void BoardSpy::initResource() {

	if (!initBitmaps())
		fatal(_T("resource bitmaps fail"));

	// detect display device
	hDisplayDC_ = CreateDC(_T("display"), NULL, NULL, NULL);
	auto ibits = GetDeviceCaps(hDisplayDC_, BITSPIXEL)*GetDeviceCaps(hDisplayDC_, PLANES);

	if (ibits <= 16) {
		fatal(_T("not supported display device"));
	}

	if (ibits <= 24)
		nDispalyBitsCount_ = 24;
	else
		nDispalyBitsCount_ = 32;

	nBpp_ = nDispalyBitsCount_ >> 3;


	boardBitmapInfo_.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	boardBitmapInfo_.bmiHeader.biWidth = tplBoardSize_; // targetWidth
	boardBitmapInfo_.bmiHeader.biHeight = tplBoardSize_; //targetHeight
	boardBitmapInfo_.bmiHeader.biPlanes = 1;
	boardBitmapInfo_.bmiHeader.biBitCount = nDispalyBitsCount_;
	boardBitmapInfo_.bmiHeader.biClrImportant = 0;
	boardBitmapInfo_.bmiHeader.biClrUsed = 0;
	boardBitmapInfo_.bmiHeader.biCompression = BI_RGB;
	boardBitmapInfo_.bmiHeader.biSizeImage = 0;
	boardBitmapInfo_.bmiHeader.biYPelsPerMeter = 0;
	boardBitmapInfo_.bmiHeader.biXPelsPerMeter = 0;

	boardDIBs_.resize(tplBoardSize_*tplBoardSize_*nBpp_);


	char program_path[1024];
	HMODULE hModule = GetModuleHandle(NULL);
	GetModuleFileNameA(hModule, program_path, 1024);
	int last = (int)strlen(program_path);
	while (last--){
		if (program_path[last] == '\\' || program_path[last] == '/') {
			program_path[last] = '\0';
			break;
		}
	}
	strcat(program_path, "\\data\\nn_config.txt");

	init(program_path);
}


bool BoardSpy::initBitmaps() {

	TCHAR resdir[256];
	TCHAR path[256];

	lstrcpy(resdir, g_szLocalPath);
	lstrcat(resdir, _T("\\res\\"));

	tplStoneSize_ = 0;

	for (auto i = 0; i < max_stone_templates; i++) {
		lstrcpy(path, resdir);
		lstrcat(path, stone_files[i].file);
		int w, h;
		if (!loadBmpData(path, stoneImages_[i], w, h))
			return false;

		if (i == 0) {
			tplStoneSize_ = w;
		}
		else {
			if (tplStoneSize_ != w || tplStoneSize_ != h)
				return false;
		}
	}

	tplBoardSize_ = tplStoneSize_ * 19;

	blackImage_.resize(tplBoardSize_*tplBoardSize_*3, 0);
	whiteImage_.resize(tplBoardSize_*tplBoardSize_ * 3, 255);

	// load stone mask
	stoneMaskData_.resize(tplStoneSize_*tplStoneSize_);
	std::fill(stoneMaskData_.begin(), stoneMaskData_.end(), 1);

	for (auto n = 0; n < sizeof(mask_files) / sizeof(LPCTSTR); ++n) {
		lstrcpy(path, resdir);
		lstrcat(path, mask_files[n]);

		std::vector<BYTE> data;
		int width, height;
		if (!loadBmpData(path, data, width, height))
			return false;

		if (width != tplStoneSize_ || height != tplStoneSize_)
			return false;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				auto idx = i*width + j;
				auto target = i*width * 3 + j * 3;
				auto r = data[target];
				auto g = data[target + 1];
				auto b = data[target + 2];

				if (r != 0 || g != 0 || b != 255) {
					stoneMaskData_[idx] = 0;
				}
			}
		}
	}

	lastMoveMaskData_.resize(tplStoneSize_*tplStoneSize_);
	std::fill(lastMoveMaskData_.begin(), lastMoveMaskData_.end(), 0);
	lstrcpy(path, resdir);
	lstrcat(path, mask_files[0]);

	std::vector<BYTE> data;
	int width, height;
	if (!loadBmpData(path, data, width, height))
		return false;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++) {
			auto idx = i*width + j;
			auto target = i*width * 3 + j * 3;
			auto r = data[target];
			auto g = data[target + 1];
			auto b = data[target + 2];

			if (r == 255 && g == 0 && b == 0) {
				lastMoveMaskData_[idx] = 1;
			}
		}
	}


	return true;
}


int BoardSpy::detectStone(int move, bool& isLastMove) const {

	isLastMove = false;

	double val[max_stone_templates];
	for (auto i = 0; i < max_stone_templates; ++i) {
		val[i] = compareBoardRegionAt(move, stoneImages_[i], stoneMaskData_);
	}

	double minval = 100000.0;
	int sel = 0;
	for (auto i = 0; i < max_stone_templates; ++i) {
		if (minval > val[i]) { minval = val[i]; sel = stone_files[i].value; }
	}

	if (sel == 1) {
		auto val2 = compareBoardRegionAt(move, whiteImage_, lastMoveMaskData_);
		if (val2 < 110)
			isLastMove = true;
	}else if (sel == -1) {
		auto val2 = compareBoardRegionAt(move, blackImage_, lastMoveMaskData_);
		if (val2 < 110)
			isLastMove = true;
	}

	return sel;
}

double BoardSpy::compareBoardRegionAt(int idx, const std::vector<BYTE>& stone, const std::vector<BYTE>& mask) const {

	auto left = tplStoneSize_*(idx%19);
	auto top = tplStoneSize_*(idx/19);

	auto bottom = top + tplStoneSize_;
	auto right = left + tplStoneSize_;

	
	auto biWidthBytes = ((tplBoardSize_ * nDispalyBitsCount_ + 31) & ~31) / 8;

	int64_t totalLast = 0;
	double total = 0;


	int counts = 0;
	const BYTE* data = &boardDIBs_[0];
	auto pmask = &mask[0];

	const BYTE* pstone = &stone[0];
	for (auto i = top; i < bottom; i++)
	{
		auto y = 25 * 19 - i - 1;
		for (int j = left; j < right; j++) {

			auto idx_m = (i - top)*tplStoneSize_ + (j - left);
			if (!pmask[idx_m]) continue;

			counts++;
			
			auto idx = y*biWidthBytes + j*nBpp_;
			auto r = data[idx + 2];
			auto g = data[idx + 1];
			auto b = data[idx];
			
			auto idx2 = (i - top)*tplStoneSize_ * 3 + (j - left) * 3;
			auto r2 = pstone[idx2];
			auto g2 = pstone[idx2 + 1];
			auto b2 = pstone[idx2 + 2];
			
			//total += abs(r - r2) + abs(g - g2) + abs(b - b2);
			total += colorDiff(r, g, b, r2, g2, b2);
		}
	}

	double diff = (double)total / (double)counts;
	return diff;
}

bool BoardSpy::scanBoard(int data[], int& lastMove) {

	if (!IsWindow(hTargetWnd))
		return false;

	// copy screen to bitmap
	if (stoneSize_ != tplStoneSize_) {

		StretchBlt(hBoardDC_, 0, 0, tplBoardSize_, tplBoardSize_, hTargetDC, offsetX_, offsetY_, stoneSize_ * 19, stoneSize_ * 19, SRCCOPY);
	}
	else {
		BitBlt(hBoardDC_, 0, 0, tplBoardSize_, tplBoardSize_, hTargetDC, offsetX_, offsetY_, SRCCOPY);
	}

	GetDIBits(hDisplayDC_, hBoardBitmap_, 0, (UINT)tplBoardSize_, &boardDIBs_[0], &boardBitmapInfo_, DIB_RGB_COLORS);

	lastMove = -1;

	for (auto idx = 0; idx < 361; idx++) {
		bool isLastMove;
		auto stone = detectStone(idx, isLastMove);
		data[idx] = stone;

		if (isLastMove)
			lastMove = idx;
	}

	return true;
}


bool BoardSpy::detecteBoard() {

	RECT rc;
	if (!GetWindowRect(hTargetWnd, &rc))
		return false;

	if (targetRect_.left != rc.left ||
		targetRect_.top != rc.top ||
		targetRect_.right != rc.right ||
		targetRect_.bottom != rc.bottom) {

		auto hWnd = hTargetWnd;

		for (auto i = 0; i < 100; ++i) {
			if (!IsWindow(hWnd))
				return false;
			if (setWindowInternal(hWnd))
				break;
			Sleep(1);
		}

		if (onSizeChanged)
			onSizeChanged();
	}

	int curBoard[361];
	int lastMove;
	if (!scanBoard(curBoard, lastMove))
		return false;

	auto thres = THRESH_STONE_EXISTS_INTERVAL;

	int sel = -1;
	std::vector<int> candicates;
	for (auto idx = 0; idx < 361; idx++) {

		auto stone = curBoard[idx];
		auto old = board[idx];

		if (old != stone) {
			board_age[idx]=0;
			board[idx] = stone;
		}
		else {
			board_age[idx]++;
			if (board_age[idx] > thres) {
				if (stone != 0 && board_last[idx]==0) {
					candicates.push_back(idx);
				}
			}
		}
	}

	if (candicates.size()) {
		auto mLast = -1;
		auto mTurn = -1;
		auto mOther = -1;
		for (auto c : candicates) {
			if (c == lastMove) mLast = c;
			else if (board[c] == turn_) mTurn = c;
			else mOther = c;
		}

		if (mTurn >= 0) sel = mTurn;
		else if (mOther >= 0) sel = mOther;
		else sel = mLast;
	}

	if (sel >= 0) {

		auto player = board[sel];
		board_last[sel] = player;

		for (int idx = 0; idx < 361; idx++) {
			if (board_age[idx] > thres && board[idx]==0) {
				board_last[idx] = 0;
			}
		}

		memcpy(board, board_last, sizeof(board));
		memset(board_age, 0, sizeof(board));

		auto x = sel % 19, y = sel / 19;
		commitMove(player, x, y);
	}
	
	return true;
}


void BoardSpy::releaseWindows() {

	if (hTargetDC) {
		ReleaseDC(hTargetWnd, hTargetDC);
		hTargetDC = NULL;
	}

	hTargetWnd = NULL;
}

bool BoardSpy::setWindowInternal(HWND hwnd) {

	RECT rc;
	if (!GetWindowRect(hwnd, &rc))
		return false;
	auto w = rc.right - rc.left;
	auto h = rc.bottom - rc.top;

	releaseWindows();

	hTargetWnd = hwnd;
	hTargetDC = GetWindowDC(hTargetWnd);

	if (!hBoardBitmap_) {

		hBoardDC_ = CreateCompatibleDC(hTargetDC);
		SetStretchBltMode(hBoardDC_, HALFTONE);

		hBoardBitmap_ = CreateCompatibleBitmap(hTargetDC, tplBoardSize_, tplBoardSize_);
		hBoardDC_.selectBitmap(hBoardBitmap_);
	}


	if (!findBoard()) {
		releaseWindows();
		return false;
	}

	return true;
}

void BoardSpy::setWindow(HWND hwnd) {
	if (setWindowInternal(hwnd))
		reAttach();
}


bool BoardSpy::findBoard() {

	RECT rc;
	if (!GetWindowRect(hTargetWnd, &rc))
		return false;

	auto width = rc.right - rc.left;
	auto height = rc.bottom - rc.top;
	targetRect_ = rc;

	HBitmap hbitmap;
	Hdc memDC;

	memDC = CreateCompatibleDC(hTargetDC);
	hbitmap = CreateCompatibleBitmap(hTargetDC, width, height);
	memDC.selectBitmap(hbitmap);

	// bitblt the window to the bitmap
	BitBlt(memDC, 0, 0, width, height, hTargetDC, 0, 0, SRCCOPY);

	OpenClipboard(NULL);
	EmptyClipboard();
	SetClipboardData(CF_BITMAP, hbitmap);
	CloseClipboard();

	// DIB DATA
	std::vector<BYTE> dib(width*height*nBpp_);
	BITMAPINFO binfo;
	binfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	binfo.bmiHeader.biWidth = width;
	binfo.bmiHeader.biHeight = height;
	binfo.bmiHeader.biPlanes = 1;
	binfo.bmiHeader.biBitCount = nDispalyBitsCount_;
	binfo.bmiHeader.biClrImportant = 0;
	binfo.bmiHeader.biClrUsed = 0;
	binfo.bmiHeader.biCompression = BI_RGB;
	binfo.bmiHeader.biSizeImage = 0;
	binfo.bmiHeader.biYPelsPerMeter = 0;
	binfo.bmiHeader.biXPelsPerMeter = 0;
	GetDIBits(hDisplayDC_, hbitmap, 0, (UINT)height, &dib[0], &binfo, DIB_RGB_COLORS);

	auto biWidthBytes = ((width * nDispalyBitsCount_ + 31) & ~31) / 8;

	auto guessSize = std::min(width, height);

	auto findRow = [&](int from) {
		for (auto _y = from; _y < height; ++_y) {
			auto y = height - _y - 1;
			auto blackPixelCount = 0;
			for (auto x = 0; x < width; ++x) {
				auto idx = y*biWidthBytes + x*nBpp_;
				auto r = dib[idx + 2];
				auto g = dib[idx + 1];
				auto b = dib[idx];

				auto diff = colorDiff(r, g, b, 0, 0, 0);
				if (diff < 50.0f) {
					blackPixelCount++;
				}
			}
			if (blackPixelCount > guessSize / 2) {
				return _y;
			}
		}
		return -1;
	};

	auto findCol = [&](int from) {
		for (auto x = from; x < width; ++x) {
			auto blackPixelCount = 0;
			for (auto _y = 0; _y < height; ++_y) {
				auto y = height - _y - 1;
				auto idx = y*biWidthBytes + x*nBpp_;
				auto r = dib[idx + 2];
				auto g = dib[idx + 1];
				auto b = dib[idx];

				auto diff = colorDiff(r, g, b, 0, 0, 0);
				if (diff < 50.0f) {
					blackPixelCount++;
				}
			}
			if (blackPixelCount > width / 2) {
				return x;
			}
		}
		return -1;
	};

	auto row0 = findRow(0);
	if (row0 < 0)
		return false;

	auto row1 = findRow(row0+10);
	if (row1 < 0)
		return false;

	auto row2 = findRow(row1 + 10);
	if (row2 < 0)
		return false;

	auto col0 = findCol(0);
	if (col0 < 0)
		return false;

	auto col1 = findCol(col0 + 10);
	if (col1 < 0)
		return false;

	auto col2 = findCol(col1 + 10);
	if (col2 < 0)
		return false;

	auto size1 = col2 - col1;
	auto size2 = row2 - row1;
	stoneSize_ = std::min(size1, size2);


	offsetX_ = col1 - stoneSize_ - (stoneSize_ +1) / 2;
	offsetY_ = row1 - stoneSize_ - (stoneSize_ +1) / 2;

	return true;
}


bool BoardSpy::reAttach() {

	int curBoard[361];
	int lastMove;
	std::vector<int> blacks;
	std::vector<int> whites;
	std::vector<int> seq;

	if (!scanBoard(curBoard, lastMove))
		return false;

	int more_stones = 0;
	int stone_counts = 0;
	for (auto idx=0; idx<361; idx++) {
		if (curBoard[idx] != 0) {
			stone_counts++;
			if (board[idx] == 0)
				more_stones++;
		}
	}

	if (more_stones > 1)
		return false;

	if (stone_counts <= 1) {
		memset(board, 0, sizeof(board));
		memset(board_last, 0, sizeof(board));
		memset(board_age, 0, sizeof(board));

		if (stone_counts == 0) {
			newGame(1);
		} else 
			newGame(-1);
	}

	return true;
}


bool BoardSpy::found() const {
	return IsWindow(hTargetWnd);
}

POINT BoardSpy::coord2Screen(int x, int y) const {

	POINT pt;
	pt.x = 0;
	pt.y = 0;

	if (!found())
		return pt;

	RECT rc;
	GetWindowRect(hTargetWnd, &rc);
	pt.y = rc.top + offsetY_ + y*stoneSize_ + stoneSize_ / 2;
	pt.x = rc.left + offsetX_ + x*stoneSize_ + stoneSize_ / 2;
	return pt;
}


void BoardSpy::placeAt(int x, int y) {

	POINT at = coord2Screen(x, y);

	POINT pt;
	GetCursorPos(&pt);
	SetCursorPos(at.x, at.y);

	mouse_event( MOUSEEVENTF_LEFTDOWN, 0, 0, 0, NULL);
	Sleep(10);
	mouse_event( MOUSEEVENTF_LEFTUP, 0, 0, 0, NULL);

	SetCursorPos(pt.x, pt.y);

	/*
	if (!found())
		return;

	int Y = offsetY_ + y*stoneSize_ + stoneSize_ / 2;
	int X = offsetX_ + x*stoneSize_ + stoneSize_ / 2;
	int lparam =  (Y << 16) +  X;
	::PostMessage(hTargetWnd, WM_LBUTTONDOWN, 0, lparam);
	::PostMessage(hTargetWnd, WM_LBUTTONUP, 0, lparam);
	*/
}


inline void trim(std::string &ss)   
{   
	auto p=find_if(ss.rbegin(),ss.rend(),std::not1(std::ptr_fun(isspace)));   
    ss.erase(p.base(),ss.end());  
	auto p2 = std::find_if(ss.begin(),ss.end(),std::not1(std::ptr_fun(isspace)));   
	ss.erase(ss.begin(), p2);
} 

void BoardSinker::init(const std::string& cfg_path) {

	std::string weights_path;
	auto pos = cfg_path.rfind('\\') + 1;
	auto base_dir = cfg_path.substr(0, pos);
	weights_path = base_dir + "leela.weights";

	std::ifstream ifs(cfg_path);
	std::string str;

	// Read line by line.
	while (ifs && getline(ifs, str)) {
		auto eq = str.find("=");
		if(eq == std::string::npos)
			continue;
		
		auto key = str.substr(0, eq);
		auto value = str.substr(eq+1);
		trim(key);
		trim(value);

		if (key == "weights") weights_path 	= base_dir + value; 
	}


	model = std::unique_ptr<ZeroPredictModel>(new ZeroPredictModel());
	if (!model->load_weights(weights_path))
	{
        fatal(_T("load model weights fail"));
    }

	agent = std::unique_ptr<AQ>(new AQ(model.get(), cfg_path));
	gtp = std::make_shared<Gtp>(agent.get());
	gtp->run();
	gtp->send_command("isready");

	std::thread([&](){
		while (true) {
			std::string prev_cmd, rsp;
			gtp->unsolicite(prev_cmd, rsp);
			if (prev_cmd == "quit")
				break;

			if (prev_cmd == "isready")
				if (onGtpReady)
					onGtpReady();

			if (prev_cmd.find("genmove") == 0) {
				// = d4
				auto movetext = rsp.substr(2);
				auto xy = movetext2xy(movetext);

				if (xy.first == 0) {
					if (onMovePass)
						onMovePass();
				} else if (xy.first == -1) {
					if (onResign)
						onResign();
				} else {
					int player = 1;
					if (prev_cmd.find("genmove w") == 0)
						player = -1;

					if (onMovePredict)
						onMovePredict(player, xy.first-1, xy.second-1);
				}
			}
		}
	}).detach();
}

void BoardSinker::deinit() {
	gtp->send_command("quit");
	gtp->join();
}

void BoardSinker::commitMove(int player, int x, int y) {

	turn_ = -player;
	steps_++;

	auto movetext = xy2movetext(x+1, y+1);
	if (player == 1)
		gtp->send_command(std::string("play b ") + movetext);
	else 
		gtp->send_command(std::string("play w ") + movetext);

	if (-player == myTurn_) {
		gtp->send_command(myTurn_ == 1 ? "genmove b nocommit" : "genmove w nocommit");
	}

	if (onMoveChange)
		onMoveChange(player, x, y, steps_);
}


void BoardSinker::newGame(int mycolor) {

	steps_ = 0;
	turn_ = 1;
	myTurn_ = mycolor;

	gtp->send_command("clear_board");
	if (mycolor == 1)
		gtp->send_command("genmove b nocommit");
}


void BoardSinker::stopThink() {
	gtp->stop_thinking();
}
