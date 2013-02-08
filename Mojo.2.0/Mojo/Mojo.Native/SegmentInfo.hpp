#pragma once

#include "Mojo.Core/D3D11.hpp"
#include "Mojo.Core/Stl.hpp"

namespace Mojo
{
namespace Native
{

struct SegmentInfo
{
    unsigned int         id;
    std::string          name;
    long                 size;
    char                 confidence;

    SegmentInfo();
    SegmentInfo( unsigned int id, const std::string& name, long size, char confidence )
        : id( id ), name( name ), size ( size ), confidence ( confidence ){}

    bool operator< (const SegmentInfo& e) const
    {
        return id < e.id;
    }
};

}
}