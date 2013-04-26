#pragma once

#ifdef _DEBUG
    
    #undef _DEBUG
    
    //#include <boost/array.hpp>
    #include <boost/filesystem.hpp>
    #include <boost/lexical_cast.hpp>

    #define _DEBUG 1

#else

    //#include <boost/array.hpp>
    #include <boost/filesystem.hpp>
    #include <boost/lexical_cast.hpp>

#endif