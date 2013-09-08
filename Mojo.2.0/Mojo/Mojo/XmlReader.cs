using System;
using System.IO;
using System.Xml;
using System.Xml.Serialization;

namespace Mojo
{
    public static class XmlReader
    {
        public static T ReadFromFile<T>( string fileName )
        {
            try
            {
                var xmlSerializer = new XmlSerializer( typeof( T ) );

                // A FileStream is needed to read the XML document.
                using ( var fileStream = new FileStream( fileName, FileMode.Open ) )
                {
                    System.Xml.XmlReader xmlTextReader = new XmlTextReader( fileStream );

                    // Declare an object variable of the type to be deserialized.
                    return (T)xmlSerializer.Deserialize( xmlTextReader );
                }
            }
            catch (Exception e)
            {
                throw new Exception( "Error reading XML from " + fileName + ":\n" + e.Message );
            }
        }
    }
}
