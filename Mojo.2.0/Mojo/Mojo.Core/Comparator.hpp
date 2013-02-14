#pragma once

#include "MojoVectors.hpp"

namespace Mojo
{
namespace Core
{

struct Int4Comparator
{
    bool operator() (const MojoInt4 l, MojoInt4 r) const
    {
        return l.w < r.w || ( l.w == r.w && (
            l.z < r.z || ( l.z == r.z && (
            l.y < r.y || ( l.y == r.y && l.x < r.x ) ) ) ) );
    }
};

struct Int3Comparator
{
    bool operator() (const MojoInt3 l, MojoInt3 r) const
    {
        return l.z < r.z || ( l.z == r.z && (
            l.y < r.y || ( l.y == r.y && l.x < r.x ) ) );
    }
};

struct Int2Comparator
{
    bool operator() (const MojoInt2 l, MojoInt2 r) const
    {
        return l.y < r.y || ( l.y == r.y && l.x < r.x );
    }
};

}
}