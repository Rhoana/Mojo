using Mojo.Interop;

namespace Mojo
{
    public class DatasetDescription
    {
        public ObservableDictionary<string, VolumeDescription> VolumeDescriptions { get; set; }
        public ObservableDictionary<int, SegmentationLabelDescription> SegmentationLabelDescriptions { get; set; }
    }
}
