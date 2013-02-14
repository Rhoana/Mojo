#include "TileCacheEntry.hpp"

namespace Mojo
{
namespace Native
{

TileCacheEntry::TileCacheEntry() :
keepState      ( TileCacheEntryKeepState_CanDiscard ),
indexTileSpace ( Mojo::Core::MojoInt4( TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX, TILE_CACHE_PAGE_TABLE_BAD_INDEX ) ),
centerDataSpace( Mojo::Core::MojoFloat3( -1.0f, -1.0f, -1.0f ) ),
extentDataSpace( Mojo::Core::MojoFloat3( -1.0f, -1.0f, -1.0f ) ),
active         ( false )
{
}

}
}
