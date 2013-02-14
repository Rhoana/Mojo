#pragma once

#include "Stl.hpp"

#define NOMINMAX
#include <Windows.h>

#include "Printf.hpp"
#include "ToString.hpp"

#define MOJO_ASSERT_HELPER( expression, modalDialogTitle )                                                                                                             \
    do                                                                                                                                                                 \
    {                                                                                                                                                                  \
        if ( !( expression ) )                                                                                                                                         \
        {                                                                                                                                                              \
            std::string assertMessage = Mojo::Core::ToString( "Filename: ", __FILE__, "\n\n\n\nLine Number: ", __LINE__, "\n\n\n\nExpression: ", #expression );        \
            Mojo::Core::Printf( "\n\n\n", assertMessage, "\n\n\n" );                                                                                                   \
            int  response   = MessageBoxA( 0, assertMessage.c_str(), modalDialogTitle, MB_ABORTRETRYIGNORE | MB_SETFOREGROUND | MB_SYSTEMMODAL | MB_ICONEXCLAMATION ); \
            bool debugBreak = false;                                                                                                                                   \
            switch( response )                                                                                                                                         \
            {                                                                                                                                                          \
                case IDABORT:  exit( -1 );                                                                                                                             \
                case IDIGNORE:                    break;                                                                                                               \
                case IDRETRY:  debugBreak = true; break;                                                                                                               \
                default:       debugBreak = true; break;                                                                                                               \
            }                                                                                                                                                          \
            if ( debugBreak )                                                                                                                                          \
            {                                                                                                                                                          \
                __debugbreak();                                                                                                                                        \
                debugBreak = false;                                                                                                                                    \
            }                                                                                                                                                          \
            while ( debugBreak )                                                                                                                                       \
            {                                                                                                                                                          \
            }                                                                                                                                                          \
        }                                                                                                                                                              \
    } while ( 0 )                                                                                                                                                      \

#define RELEASE_ASSERT( expression ) MOJO_ASSERT_HELPER( expression, "RELEASE_ASSERT" )

#ifdef _DEBUG
    #define ASSERT( expression ) MOJO_ASSERT_HELPER( expression, "ASSERT" )
#else
    #define ASSERT( expression )
#endif