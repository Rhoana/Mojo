using System;
using System.Diagnostics;

namespace Mojo
{
    public static class Console
    {
        public static void WriteLine( string outputString, params object[] args )
        {
            System.Console.WriteLine( outputString, args );
            Trace.WriteLine( String.Format( outputString, args ) );
        }
    }
}
