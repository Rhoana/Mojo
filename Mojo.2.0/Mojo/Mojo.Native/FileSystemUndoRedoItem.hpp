#pragma once

#include <time.h>
#include <bitset>

#include "Mojo.Core/Comparator.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"
#include "FileSystemIdIndex.hpp"
#include "Constants.hpp"

namespace Mojo
{
namespace Native
{

struct FileSystemUndoRedoItem
{
    FileSystemUndoRedoItem();

    int                                                                 oldId;
    int                                                                 newId;
    FileSystemTileSet                                                   idTileMapAddNewId;

	Core::HashMap< int, FileSystemTileSet >                             idTileMapRemoveOldIdSets;

	Core::HashMap< std::string,
		std::bitset< TILE_SIZE * TILE_SIZE > >
		                                                                changePixels;
	Core::HashMap< std::string,
		std::set< int2, Core::Int2Comparator > >
		                                                                changeSets;
};

}
}