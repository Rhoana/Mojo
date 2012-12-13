#include "FileSystemUndoRedoItem.hpp"

namespace Mojo
{
namespace Native
{

    FileSystemUndoRedoItem::FileSystemUndoRedoItem() :
		oldId                 ( 0 ),
		newId                 ( 0 )
		{
			changePixels = Core::HashMap< std::string, std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > >();
		}

}
}