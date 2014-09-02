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
    int                  confidence;
    bool                 changed;
    std::string          type;
    std::string          subtype;

    SegmentInfo();

    SegmentInfo( unsigned int id, const std::string& name, long size, int confidence, std::string& type, std::string& subtype )
        : id( id ), name( name ), size ( size ), confidence ( confidence ), type ( type ), subtype( subtype ), changed ( false ){}

    SegmentInfo( unsigned int id, const std::string& name, long size, int confidence, std::string& type, std::string& subtype, bool changed )
    : id( id ), name( name ), size ( size ), confidence ( confidence ), type ( type ), subtype( subtype ), changed ( changed ){}

    bool operator< (const SegmentInfo& e) const
    {
        return id < e.id;
    }
};

}
}