#pragma once

#include "Cuda.hpp"

namespace Mojo
{
namespace Core
{

struct Int4Comparator
{
    bool operator() (const int4 l, int4 r) const
    {
        return l.w < r.w || ( l.w == r.w && (
            l.z < r.z || ( l.z == r.z && (
            l.y < r.y || ( l.y == r.y && l.x < r.x ) ) ) ) );
    }
};

struct Int3Comparator
{
    bool operator() (const int3 l, int3 r) const
    {
        return l.z < r.z || ( l.z == r.z && (
            l.y < r.y || ( l.y == r.y && l.x < r.x ) ) );
    }
};

struct Int2Comparator
{
    bool operator() (const int2 l, int2 r) const
    {
        return l.y < r.y || ( l.y == r.y && l.x < r.x );
    }
};

}
}