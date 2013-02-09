#pragma once

#include "Mojo.Native/SegmentInfo.hpp"

using namespace System;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class SegmentInfo
{
public:
    SegmentInfo();
    SegmentInfo( Native::SegmentInfo segmentInfo, std::string color );

    Native::SegmentInfo ToNative();

    property unsigned int  Id;
    property String^       Name;
    property long          Size;
    property int           Confidence;
	property String^       Color;

};

}
}
