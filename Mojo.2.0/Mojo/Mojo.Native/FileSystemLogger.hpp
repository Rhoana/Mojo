#pragma once

//#include <boost\filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

namespace Mojo
{
namespace Native
{

class FileSystemLogger
{

public:
    FileSystemLogger();

    void                                              OpenLog( std::string logFilePath );
    void                                              CloseLog();

    void                                              Log( std::string messages );
                                               
private:                                       

    std::string                                       mLogFilePath;
    bool                                              mIsLogFileOpen;
    boost::filesystem::ofstream                       mLogFileStream;

};

}
}