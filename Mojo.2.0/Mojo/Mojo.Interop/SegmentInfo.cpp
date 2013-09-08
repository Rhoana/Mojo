#include "SegmentInfo.hpp"

#include <msclr/marshal_cppstd.h>

namespace Mojo
{
namespace Interop
{

SegmentInfo::SegmentInfo()
{
}

SegmentInfo::SegmentInfo( Native::SegmentInfo segmentInfo, std::string colorString )
{
    Id                 = segmentInfo.id;
    Name               = msclr::interop::marshal_as< String^ >( segmentInfo.name );
    Size               = segmentInfo.size;
    Confidence         = segmentInfo.confidence;
    Color              = msclr::interop::marshal_as< String^ >( colorString );
}

Native::SegmentInfo SegmentInfo::ToNative()
{
    Native::SegmentInfo segmentInfo;

    segmentInfo.id          = Id;
    segmentInfo.name        = msclr::interop::marshal_as< std::string >( Name );
    segmentInfo.size        = Size;
    segmentInfo.confidence  = Confidence;

    return segmentInfo;

}

}
}