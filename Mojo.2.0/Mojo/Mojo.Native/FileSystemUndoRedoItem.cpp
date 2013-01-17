#include "FileSystemUndoRedoItem.hpp"

namespace Mojo
{
namespace Native
{

    FileSystemUndoRedoItem::FileSystemUndoRedoItem() :
		oldId                 ( 0 ),
		newId                 ( 0 )
		{
			changePixels = Core::HashMap< std::string, std::bitset< TILE_SIZE * TILE_SIZE > >();
		}

}
}