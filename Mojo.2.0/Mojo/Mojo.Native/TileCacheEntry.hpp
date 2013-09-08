#pragma once

#include "Types.hpp"
#include "HashMap.hpp"
#include "ID3D11Texture.hpp"

namespace Mojo
{
namespace Native
{

const int TILE_CACHE_BAD_INDEX            = -1;
const int TILE_CACHE_PAGE_TABLE_BAD_INDEX = -2;

struct TileCacheEntry
{
    TileCacheEntry();

    HashMap< std::string, ID3D11Texture* > d3d11Textures;
    Int4                                   indexTileSpace;
    Float3                                 centerDataSpace;
    Float3                                 extentDataSpace;
    bool                                   active;
};

}
}