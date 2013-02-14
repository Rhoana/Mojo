#include "Printf.hpp"

#include "Stl.hpp"

#include <stdio.h>

#define NOMINMAX
#include <Windows.h>

#include "ToString.hpp"

namespace Mojo
{
namespace Core
{

void PrintfHelper( std::string string )
{
    std::cout << string << std::endl;
    OutputDebugString( ToString( string + "\n" ).c_str() );

}

}
}

