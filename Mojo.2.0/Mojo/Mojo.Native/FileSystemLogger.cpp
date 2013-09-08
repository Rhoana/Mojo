#include "FileSystemLogger.hpp"

#include "Printf.hpp"
#include "Boost.hpp"

namespace Mojo
{
namespace Native
{

FileSystemLogger::FileSystemLogger(): mLogFileStream ()
{
    mLogFilePath = "";
    mIsLogFileOpen = false;
}

void FileSystemLogger::CloseLog()
{
    if ( mIsLogFileOpen )
    {
        Log( "Closing log file.");
        mLogFileStream.close();
        mIsLogFileOpen = false;
    }
}

void FileSystemLogger::OpenLog( std::string logFilePath )
{
    if ( mIsLogFileOpen )
    {
        CloseLog();
    }

    mLogFilePath = logFilePath;

    boost::filesystem::path logPath = boost::filesystem::path( mLogFilePath );
    if ( !boost::filesystem::exists( logPath ) )
    {
        boost::filesystem::create_directories( logPath.parent_path() );
    }

    mLogFileStream.open( logPath, std::ios::app );

    mIsLogFileOpen = true;

    Log( "Log file opened.");

}

void FileSystemLogger::Log( std::string message )
{
    if ( mIsLogFileOpen )
    {
        std::time_t t = std::time( NULL );

        // C++ ISO Compliant
        //std::tm tm = *std::localtime( &t );

        // Microsoft Compliant
        std::tm tm;
        localtime_s( &tm, &t );

        mLogFileStream << std::put_time( &tm, "%Y:%m:%d-%H:%M:%S: " ) << message << std::endl;
    }
    else
    {
        Printf( "WARNING: Log file not open. Could not log message:" );
        Printf( message );
    }
}


}
}
