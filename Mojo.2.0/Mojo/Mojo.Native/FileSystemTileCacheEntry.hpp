#pragma once

#include "Stl.hpp"
#include "HashMap.hpp"
#include "VolumeDescription.hpp"
#include "Types.hpp"

namespace Mojo
{
namespace Native
{

struct FileSystemTileCacheEntry
{
    FileSystemTileCacheEntry();

    Int4                                      tileIndex;
    int                                       inUse;
    clock_t                                   timeStamp;
    bool                                      needsSaving;
    HashMap< std::string, VolumeDescription > volumeDescriptions;
};

}
}