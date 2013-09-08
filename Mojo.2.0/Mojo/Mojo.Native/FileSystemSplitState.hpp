#pragma once

#include "Stl.hpp"
#include "Types.hpp"

namespace Mojo
{
namespace Native
{

struct FileSystemSplitState
{
    FileSystemSplitState();

    int                                      splitId;
    int                                      splitZ;
    std::vector< Float2 >                    splitLine;
    std::vector< std::pair< Float2, char > > splitDrawPoints;

};

}
}