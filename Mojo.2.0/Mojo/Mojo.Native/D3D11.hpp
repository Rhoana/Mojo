#pragma once

#include "Stl.hpp"

#include <d3d11.h>

#include <DXGIFormat.h>

#include "Assert.hpp"
#include "Printf.hpp"

#define MOJO_D3D_CHECK_RESULT_DEBUG( d3dResult )                                         \
    do                                                                                   \
    {                                                                                    \
        ASSERT( d3dResult != E_FAIL                                                   ); \
        ASSERT( d3dResult != E_HANDLE                                                 ); \
        ASSERT( d3dResult != E_INVALIDARG                                             ); \
        ASSERT( d3dResult != E_OUTOFMEMORY                                            ); \
        ASSERT( d3dResult != E_POINTER                                                ); \
        ASSERT( d3dResult != D3D11_ERROR_FILE_NOT_FOUND                               ); \
        ASSERT( d3dResult != D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS                ); \
        ASSERT( d3dResult != D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS                 ); \
        ASSERT( d3dResult != D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD ); \
        ASSERT( d3dResult != S_FALSE                                                  ); \
        ASSERT( d3dResult == S_OK                                                     ); \
    } while ( 0 )


#define MOJO_D3D_CHECK_RESULT_RELEASE( d3dResult )                                               \
    do                                                                                           \
    {                                                                                            \
        RELEASE_ASSERT( d3dResult != E_FAIL                                                   ); \
        RELEASE_ASSERT( d3dResult != E_HANDLE                                                 ); \
        RELEASE_ASSERT( d3dResult != E_INVALIDARG                                             ); \
        RELEASE_ASSERT( d3dResult != E_OUTOFMEMORY                                            ); \
        RELEASE_ASSERT( d3dResult != E_POINTER                                                ); \
        RELEASE_ASSERT( d3dResult != D3D11_ERROR_FILE_NOT_FOUND                               ); \
        RELEASE_ASSERT( d3dResult != D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS                ); \
        RELEASE_ASSERT( d3dResult != D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS                 ); \
        RELEASE_ASSERT( d3dResult != D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD ); \
        RELEASE_ASSERT( d3dResult != S_FALSE                                                  ); \
        RELEASE_ASSERT( d3dResult == S_OK                                                     ); \
    } while ( 0 )

#ifdef _DEBUG
    #define MOJO_D3D_SAFE_DEBUG( functionCall )       \
        do                                            \
        {                                             \
            HRESULT d3dResult = functionCall ;        \
            MOJO_D3D_CHECK_RESULT_DEBUG( d3dResult ); \
        } while ( 0 )
#else
    #define MOJO_D3D_SAFE_DEBUG( functionCall ) functionCall
#endif


#define MOJO_D3D_SAFE_RELEASE( functionCall )       \
    do                                              \
    {                                               \
        HRESULT d3dResult = functionCall ;          \
        MOJO_D3D_CHECK_RESULT_RELEASE( d3dResult ); \
    } while ( 0 )


#define MOJO_D3D_SAFE( functionCall ) MOJO_D3D_SAFE_DEBUG( functionCall )
