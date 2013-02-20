#include "FileSystemUndoRedoItem.hpp"

namespace Mojo
{
namespace Native
{

    FileSystemUndoRedoItem::FileSystemUndoRedoItem() :
		oldId                 ( 0 ),
		newId                 ( 0 )
		{
			//changePixels = Core::HashMap< std::string, std::bitset< TILE_SIZE * TILE_SIZE > >();
            //changeSets = Core::HashMap< std::string, std::set< Core::MojoInt2, Core::Int2Comparator > >();
		}

}
}