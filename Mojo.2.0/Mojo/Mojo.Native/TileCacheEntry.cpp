#include "TileCacheEntry.hpp"

namespace Mojo
{
namespace Native
{

TileCacheEntry::TileCacheEntry() :
indexTileSpace ( Int4( TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX ) ),
centerDataSpace( Float3( -1.0f, -1.0f, -1.0f ) ),
extentDataSpace( Float3( -1.0f, -1.0f, -1.0f ) ),
active         ( false )
{
}

}
}
