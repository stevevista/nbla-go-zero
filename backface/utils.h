#pragma once
#include <algorithm>
#include <windows.h>

class Lock {
	CRITICAL_SECTION cs_;
public:
	Lock() {
		InitializeCriticalSection(&cs_);
	}
	~Lock() {
		DeleteCriticalSection(&cs_);
	}
	void lock() {
		EnterCriticalSection(&cs_);
	}
	void unlock() {
		LeaveCriticalSection(&cs_);
	}
};

class Hdc {
	HDC h_;
	HBITMAP bmOld_;
public:
	Hdc() : h_(NULL), bmOld_(NULL) {}
	~Hdc() {
		if (h_) {
			if (bmOld_) SelectObject(h_, bmOld_);
			DeleteDC(h_);
		}
	}

	operator bool() const { return h_ != NULL; }
	operator HDC() { return h_; }
	Hdc& operator=(HDC h) {

		if (h_) {
			if (bmOld_) SelectObject(h_, bmOld_);
			DeleteDC(h_);
		}
		h_ = h;
		bmOld_ = NULL;
		return *this;
	}

	void selectBitmap(HBITMAP bmp) {
		auto old = SelectObject(h_, bmp);
		if (bmOld_ == NULL)
			bmOld_ = (HBITMAP)old;
	}
};

class HBitmap {
	HBITMAP h_;
public:
	HBitmap() :h_(NULL) {}
	~HBitmap() {
		if (h_) DeleteObject(h_);
	}
	operator bool() const { return h_ != NULL; }
	operator HBITMAP() { return h_; }
	HBitmap& operator=(HBITMAP h) {

		if (h_) DeleteObject(h_);
		h_ = h;
		return *this;
	}

};


