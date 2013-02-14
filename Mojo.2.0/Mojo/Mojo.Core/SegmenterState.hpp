#pragma once

#include "Stl.hpp"

#include "HashMap.hpp"
#include "PrimitiveMap.hpp"
#include "HostVectorMap.hpp"
//#include "DeviceVectorMap.hpp"
#include "D3D11CudaTextureMap.hpp"
#include "VolumeDescription.hpp"

struct ID3D11Device;
struct ID3D11DeviceContext;

namespace Mojo
{
namespace Core
{

class ID3D11CudaTexture;
class TileManager;

class SegmenterState
{
public:
    SegmenterState();

    D3D11CudaTextureMap                d3d11CudaTextures;
    //DeviceVectorMap                    deviceVectors;
    HostVectorMap                      hostVectors;
    PrimitiveMap                       constParameters;
    PrimitiveMap                       dynamicParameters;

    VolumeDescription                  volumeDescription;

    HashMap< std::string, cudaArray* > cudaArrays;
    HashMap< int, float >              minCostsPerSlice;

    std::set< int >                    slicesWithForegroundConstraints;
    std::set< int >                    slicesWithBackgroundConstraints;

    ID3D11Device*                      d3d11Device;
    ID3D11DeviceContext*               d3d11DeviceContext;
};

}
}