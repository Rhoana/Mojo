namespace Mojo
{
    public interface ITool : IUserInputHandler
    {
        void Select();
        void SelectSegment( uint segmentId );
        void MoveZ();
    }
}