#pragma once

#include <time.h>
#include <bitset>
#include <functional>

#include "Types.hpp"
#include "HashMap.hpp"
#include "VolumeDescription.hpp"
#include "Types.hpp"
#include "Boost.hpp"
#include "FileSystemSegmentInfoManager.hpp"
#include "FileSystemTileServerConstants.hpp"

namespace Mojo
{
namespace Native
{

typedef std::bitset < TILE_PIXELS * TILE_PIXELS > TileChangeBits;

typedef std::map<
    unsigned int,
    TileChangeBits,
    std::less<unsigned int>,
    boost::fast_pool_allocator <
    std::pair< unsigned int, TileChangeBits > > > TileChangeIdMap;

typedef std::map<
    Int4,
    TileChangeIdMap,
    Int4Comparator >                              TileChangeMap;

struct FileSystemUndoRedoItem
{
    FileSystemUndoRedoItem();

    unsigned int                                newId;
                                                
    TileChangeMap                               tileChangeMap;
    std::map< unsigned int, long >              remapFromIdsAndSizes;
                                                
    FileSystemTileSet                           idTileMapAddNewId;
    std::map< unsigned int, FileSystemTileSet > idTileMapRemoveOldIdSets;
                                                               

};

}
}