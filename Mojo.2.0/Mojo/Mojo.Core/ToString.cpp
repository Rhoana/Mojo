#include "ToString.hpp"

#include "Stl.hpp"

#include "Boost.hpp"

namespace Mojo
{
namespace Core
{

std::string ToStringHelper( std::string x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( const char* x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( float x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( double x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( unsigned int x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( int x )
{
    return boost::lexical_cast< std::string >( x );
}

std::string ToStringHelper( std::set< unsigned int > x )
{
    std::ostringstream stringStream;
	stringStream << "[";

	if ( x.begin() == x.end() )
	{
		stringStream << " empty ";
	}
	else
	{
		stringStream << *x.begin();
	}

	for ( std::set< unsigned int >::iterator sit = ++x.begin(); sit != x.end(); ++sit )
	{
		stringStream << ",";
		stringStream << *sit;
	}

	stringStream << "]";
	return stringStream.str();
}

std::string ToStringZeroPad( int x, int totalNumChars )
{
    std::ostringstream stringStream;
    stringStream << std::setw( totalNumChars ) << std::setfill( '0' ) << x;
    return stringStream.str();
}

std::string ToStringZeroPad( unsigned int x, int totalNumChars )
{
    std::ostringstream stringStream;
    stringStream << std::setw( totalNumChars ) << std::setfill( '0' ) << x;
    return stringStream.str();
}



}

}
