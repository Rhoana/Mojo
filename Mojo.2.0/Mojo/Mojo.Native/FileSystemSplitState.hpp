#pragma once

#include "Mojo.Core/Stl.hpp"
//#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/MojoVectors.hpp"

using namespace Mojo::Core;

namespace Mojo
{
namespace Native
{

struct FileSystemSplitState
{
    FileSystemSplitState();

    int                                                           splitId;
    int                                                           splitZ;
    std::vector< MojoFloat2 >                                         splitLine;
    std::vector< std::pair< MojoFloat2, char >>                       splitDrawPoints;

};

}
}