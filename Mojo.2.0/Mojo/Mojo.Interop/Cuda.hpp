#pragma once

#using <SlimDX.dll>

using namespace SlimDX::Direct3D11;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class Cuda
{
public:
    static void Initialize( Device^ d3d11Device );
    static void Terminate();

private:
    Cuda();
};

}
}