using Mojo.Interop;
using SlimDX.DXGI;

namespace Mojo
{
    public class SegmenterImageStackLoadDescription
    {        
        public ObservableDictionary< string, string > Directories { get; set; }
        public int NumBytesPerVoxel { get; set; }
        public bool IsSigned { get; set; }
        public Format DxgiFormat { get; set; }
    }
}
