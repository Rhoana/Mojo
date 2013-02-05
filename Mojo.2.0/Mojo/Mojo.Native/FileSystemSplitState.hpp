#pragma once

#include "Mojo.Core/Stl.hpp"
#include "Mojo.Core/Cuda.hpp"

namespace Mojo
{
namespace Native
{

struct FileSystemSplitState
{
    FileSystemSplitState();

    int                                                           splitId;
    int                                                           splitZ;
    std::vector< float2 >                                         splitLine;
    std::vector< std::pair< float2, char >>                       splitDrawPoints;

};

}
}