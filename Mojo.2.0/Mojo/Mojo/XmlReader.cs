using System;
using System.IO;
using System.Xml;
using System.Xml.Serialization;

namespace Mojo
{
    public static class XmlReader
    {
        public static TXml ReadFromFile< TXml, TXmlSerializer >( string fileName ) where TXmlSerializer : XmlSerializer, new()
        {
            try
            {
                var xmlSerializer = new TXmlSerializer();

                // A FileStream is needed to read the XML document.
                using ( var fileStream = new FileStream( fileName, FileMode.Open ) )
                {
                    System.Xml.XmlReader xmlTextReader = new XmlTextReader( fileStream );

                    // Declare an object variable of the type to be deserialized.
                    return (TXml)xmlSerializer.Deserialize( xmlTextReader );
                }
            }
            catch ( Exception e )
            {
                Console.WriteLine( e.Message );
                Release.Assert( false );
            }

            return default( TXml );
        }
    }
}
