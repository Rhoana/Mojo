#include "Printf.hpp"

#include "Stl.hpp"

#include <Windows.h>

#include "ToString.hpp"

namespace Mojo
{
namespace Native
{

void PrintfHelper( std::string string )
{
    std::cout << string << std::endl;
    OutputDebugString( ToString( string + "\n" ).c_str() );

}

}
}

