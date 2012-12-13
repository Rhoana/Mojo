#include "Cuda.hpp"

#include "Mojo.Core/Cuda.hpp"

namespace Mojo
{
namespace Interop
{

Cuda::Cuda()
{
}

void Cuda::Initialize( Device^ d3d11Device )
{
    Core::Cuda::Initialize( reinterpret_cast< ID3D11Device* >( d3d11Device->ComPointer.ToPointer() ) );
}

void Cuda::Terminate()
{
    Core::Cuda::Terminate();
}

}
}