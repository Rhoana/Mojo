#pragma once

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class Thrust
{
public:
    static void Initialize();
    static void Terminate();

private:
    Thrust();
};

}
}