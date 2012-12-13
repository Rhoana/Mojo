#include "TileCacheEntry.hpp"

namespace Mojo
{
namespace Native
{

TileCacheEntry::TileCacheEntry() :
keepState      ( TileCacheEntryKeepState_CanDiscard ),
indexTileSpace ( make_int4( TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX ) ),
centerDataSpace( make_float3( -1.0f, -1.0f, -1.0f ) ),
extentDataSpace( make_float3( -1.0f, -1.0f, -1.0f ) ),
active         ( false )
{
}

}
}
