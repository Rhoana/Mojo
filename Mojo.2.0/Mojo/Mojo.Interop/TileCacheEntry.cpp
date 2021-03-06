#include "TileCacheEntry.hpp"

#include "Mojo.Native/TileCacheEntry.hpp"

namespace Mojo
{
namespace Interop
{

TileCacheEntry::TileCacheEntry()
{
    D3D11CudaTextures = gcnew ObservableDictionary< String^, SlimDX::Direct3D11::ShaderResourceView^ >();
    KeepState         = Mojo::Interop::TileCacheEntryKeepState::TileCacheEntryKeepState_CanDiscard;
    IndexTileSpace    = Vector4( (float)Native::TILE_CACHE_PAGE_TABLE_BAD_INDEX, (float)Native::TILE_CACHE_PAGE_TABLE_BAD_INDEX, (float)Native::TILE_CACHE_PAGE_TABLE_BAD_INDEX, (float)Native::TILE_CACHE_PAGE_TABLE_BAD_INDEX );
    CenterDataSpace   = Vector3( -1.0f, -1.0f, -1.0f );
    ExtentDataSpace   = Vector3( -1.0f, -1.0f, -1.0f );
    Active            = false;


}

}
}