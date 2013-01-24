using SlimDX;

namespace Mojo
{
    public class SegmentationLabelDescription
    {
        public int Id { get; set; }
        public Vector3 Color { get; set; }
        public string Name { get; set; }
        public int Size { get; set; }

        private SegmentationLabelDescription() {}

        public SegmentationLabelDescription( int id )
        {
            Id = id;
        }
    }
}
