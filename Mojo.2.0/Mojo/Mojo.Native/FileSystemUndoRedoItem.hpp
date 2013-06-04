#pragma once

#include <time.h>
#include <bitset>
#include <functional>

#include "Mojo.Core/Comparator.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"
#include "Mojo.Core/MojoVectors.hpp"
#include "Mojo.Core/Boost.hpp"
#include "FileSystemSegmentInfoManager.hpp"
#include "Constants.hpp"

using namespace Mojo::Core;

namespace Mojo
{
namespace Native
{

typedef std::bitset < TILE_PIXELS * TILE_PIXELS >              TileChangeBits;

typedef std::map<
	unsigned int,
	TileChangeBits,
	std::less<unsigned int>,
	boost::fast_pool_allocator <
	std::pair< unsigned int, TileChangeBits > > >              TileChangeIdMap;

typedef std::map<
	MojoInt4,
	TileChangeIdMap,
	Core::Int4Comparator >                                     TileChangeMap;

struct FileSystemUndoRedoItem
{
    FileSystemUndoRedoItem();

    unsigned int                                               newId;
															   
	TileChangeMap                                              tileChangeMap;
	std::map< unsigned int, long >                             remapFromIdsAndSizes;
															   
    FileSystemTileSet                                          idTileMapAddNewId;
	std::map< unsigned int, FileSystemTileSet >                idTileMapRemoveOldIdSets;
															   

};

}
}