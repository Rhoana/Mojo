#pragma once

#include <time.h>

#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"
#include "Mojo.Core/MojoVectors.hpp"

using namespace Mojo::Core;

namespace Mojo
{
namespace Native
{

struct FileSystemTileCacheEntry
{
    FileSystemTileCacheEntry();

    MojoInt4                                                 tileIndex;
    int                                                      inUse;
    clock_t                                                  timeStamp;
    bool                                                     needsSaving;
    Core::HashMap< std::string, Core::VolumeDescription >    volumeDescriptions;
};

}
}