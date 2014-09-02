#pragma once

#include "Mojo.Native/SegmentInfo.hpp"

#include "NotifyPropertyChanged.hpp"

#using <SlimDX.dll>

using namespace System;
using namespace SlimDX;
using namespace SlimDX::DXGI;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class SegmentInfo : public NotifyPropertyChanged
{
public:
    SegmentInfo();
    SegmentInfo( Native::SegmentInfo segmentInfo, std::string color );

    Native::SegmentInfo ToNative();

    property unsigned int  Id
	{
		unsigned int get() { return mId; }
		void set(unsigned int value) { mId = value; OnPropertyChanged( "Id" ); }
	}

    property String^       Name
	{
		String^ get() { return mName; }
		void set(String^ value) { mName = value; OnPropertyChanged( "Name" ); }
	}

    property long          Size
	{
		long get() { return mSize; }
		void set(long value) { mSize = value; OnPropertyChanged( "Size" ); }
	}

    property int           Confidence
	{
		int get() { return mConfidence; }
		void set(int value) { mConfidence = value; OnPropertyChanged( "Confidence" ); }
	}

	property String^       Color
	{
		String^ get() { return mColor; }
		void set(String^ value) { mColor = value; OnPropertyChanged( "Color" ); }
	}

	property String^       Type
	{
		String^ get() { return mType; }
		void set(String^ value) { mType = value; OnPropertyChanged( "Type" ); }
	}

	property String^       SubType
	{
		String^ get() { return mSubType; }
		void set(String^ value) { mSubType = value; OnPropertyChanged( "SubType" ); }
	}

private:
	unsigned int mId;
    String^       mName;
    long          mSize;
    int           mConfidence;
	String^       mColor;
	String^       mType;
	String^       mSubType;

};

}
}
