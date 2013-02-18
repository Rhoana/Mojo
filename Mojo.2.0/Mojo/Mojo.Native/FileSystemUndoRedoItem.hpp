#pragma once

#include <time.h>
#include <bitset>

#include "Mojo.Core/Comparator.hpp"
#include "Mojo.Core/HashMap.hpp"
#include "Mojo.Core/VolumeDescription.hpp"
#include "Mojo.Core/MojoVectors.hpp"
#include "FileSystemSegmentInfoManager.hpp"
#include "Constants.hpp"

namespace Mojo
{
namespace Native
{

typedef std::set< Core::MojoInt2, Core::Int2Comparator > UndoRedoChangeSet;

struct FileSystemUndoRedoItem
{
    FileSystemUndoRedoItem();

    unsigned int                                                        oldId;
    unsigned int                                                        newId;

	std::map< unsigned int, long >                                      remapFromIdsAndSizes;

    FileSystemTileSet                                                   idTileMapAddNewId;
	Core::HashMap< unsigned int, FileSystemTileSet >                    idTileMapRemoveOldIdSets;

	//Core::HashMap< std::string,
	//	std::bitset< TILE_SIZE * TILE_SIZE > >                          changePixels;

	Core::HashMap< std::string, UndoRedoChangeSet >                     changeSets;

};

}
}