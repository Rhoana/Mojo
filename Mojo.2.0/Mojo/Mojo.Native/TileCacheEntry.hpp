#pragma once

#include "Mojo.Core/D3D11.hpp"
//#include "Mojo.Core/Cuda.hpp"
#include "Mojo.Core/D3D11CudaTextureMap.hpp"
//#include "Mojo.Core/DeviceVectorMap.hpp"
#include "Mojo.Core/MojoVectors.hpp"

namespace Mojo
{
namespace Native
{

const int TILE_CACHE_BAD_INDEX            = -1;
const int TILE_CACHE_PAGE_TABLE_BAD_INDEX = -2;

enum TileCacheEntryKeepState
{
    TileCacheEntryKeepState_MustKeep,
    TileCacheEntryKeepState_CanDiscard
};

struct TileCacheEntry
{
    TileCacheEntry();

    Core::D3D11CudaTextureMap d3d11CudaTextures;
    //Core::DeviceVectorMap     deviceVectors;
    TileCacheEntryKeepState   keepState;
    Mojo::Core::MojoInt4                      indexTileSpace;
    Mojo::Core::MojoFloat3                    centerDataSpace;
    Mojo::Core::MojoFloat3                    extentDataSpace;
    bool                      active;
};

}
}