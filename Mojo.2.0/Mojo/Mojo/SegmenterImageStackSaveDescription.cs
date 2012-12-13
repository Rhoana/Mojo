using Mojo.Interop;

namespace Mojo
{
    public class SegmenterImageStackSaveDescription
    {        
        public ObservableDictionary< string, string > Directories { get; set; }
        public ObservableDictionary< string, VolumeDescription > VolumeDescriptions { get; set; }
    }
}
