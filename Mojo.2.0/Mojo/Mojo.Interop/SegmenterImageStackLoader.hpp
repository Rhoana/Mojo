#pragma once

#include "VolumeDescription.hpp"

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class SegmenterImageStackLoader
{
public:
    static void SaveIdImages( VolumeDescription^ volumeDescription, String^ path );
};

}
}