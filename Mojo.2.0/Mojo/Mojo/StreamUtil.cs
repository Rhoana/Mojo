using System.IO;

namespace Mojo
{
    public static class StreamUtil
    {
        private const long DEFAULT_STREAM_CHUNK_SIZE = 0x1000;

        public static void CopyTo( this Stream from, Stream to )
        {
            if ( !from.CanRead || !to.CanWrite )
            {
                return;
            }

            var buffer = from.CanSeek
               ? new byte[ from.Length ]
               : new byte[ DEFAULT_STREAM_CHUNK_SIZE ];
            int read;

            while ( ( read = from.Read( buffer, 0, buffer.Length ) ) > 0 )
            {
                to.Write( buffer, 0, read );
            }
        }
    }
}
