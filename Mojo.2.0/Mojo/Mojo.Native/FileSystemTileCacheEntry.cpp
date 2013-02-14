#include "FileSystemTileCacheEntry.hpp"
#include "TileCacheEntry.hpp"

namespace Mojo
{
namespace Native
{

    FileSystemTileCacheEntry::FileSystemTileCacheEntry() :
        tileIndex          ( MojoInt4( TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX ) ),
        inUse              ( 0 ),
        timeStamp          ( 0 ),
        needsSaving        ( false ),
        volumeDescriptions ()
    {
    }

}
}
 