using System.Diagnostics;
using System.Linq;

namespace Mojo
{
    public static class Debug
    {
        public static void Assert( bool condition )
        {
            if ( !condition )
            {
                new StackTrace().GetFrames().ToList().ForEach( stackFrame => Console.WriteLine( stackFrame.GetMethod().Name ) );
                if ( Debugger.IsAttached )
                {
                    Debugger.Break();
                }
            }

            System.Diagnostics.Debug.Assert( condition );
        }

        public static void Assert( bool condition, string message )
        {
            if ( !condition )
            {
                new StackTrace().GetFrames().ToList().ForEach( stackFrame => Console.WriteLine( stackFrame.GetMethod().Name ) );
                if ( Debugger.IsAttached )
                {
                    Debugger.Break();
                }
            }

            System.Diagnostics.Debug.Assert( condition, message );
        }
    }

    public static class Release
    {
        public static void Assert( bool condition )
        {
            if ( !condition )
            {
                new StackTrace().GetFrames().ToList().ForEach( stackFrame => Console.WriteLine( stackFrame.GetMethod().Name ) );
                if ( Debugger.IsAttached )
                {
                    Debugger.Break();
                }
            }

            Trace.Assert( condition );
        }

        public static void Assert( bool condition, string message )
        {
            if ( !condition )
            {
                new StackTrace().GetFrames().ToList().ForEach( stackFrame => Console.WriteLine( stackFrame.GetMethod().Name ) );
                if ( Debugger.IsAttached )
                {
                    Debugger.Break();
                }
            }

            Trace.Assert( condition, message );
        }
    }
}
