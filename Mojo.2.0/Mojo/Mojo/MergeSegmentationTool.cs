using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    public class MergeSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;

        public MergeSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
        }

        public override void SelectSegment( uint segmentId )
        {
            if ( mTileManager.SelectedSegmentId != segmentId )
            {
                mTileManager.SelectedSegmentId = segmentId;
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {
            base.OnKeyDown( keyEventArgs, width, height );

            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.Q:
                    mTileManager.ToggleShowBoundaryLines();
                    break;
                case System.Windows.Input.Key.A:
                    mTileManager.ToggleShowSegmentation();
                    break;
                case System.Windows.Input.Key.E:
                    mTileManager.IncreaseSegmentationVisibility();
                    break;
                case System.Windows.Input.Key.D:
                    mTileManager.DecreaseSegmentationVisibility();
                    break;
                case System.Windows.Input.Key.Z:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                        mTileManager.Internal.UndoChange();
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                        mTileManager.Internal.RedoChange();
                    break;
                case System.Windows.Input.Key.Escape:
                    //Unselect this segment
                    mTileManager.SelectedSegmentId = 0;
                    break;

            }
        }

        public override void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mTileManager.SegmentationLoaded )
            {
                //Get the id of the segment being clicked

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float) mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float) mouseEventArgs.Y / height ) * extentDataSpace.Y;

                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                var clickedId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                if ( mouseEventArgs.Button == MouseButtons.Left )
                {
                    //Select this segment
                    if ( clickedId > 0 )
                    {
                        if ( clickedId != mTileManager.SelectedSegmentId )
                        {
                            //Select this segment
                            mTileManager.SelectedSegmentId = clickedId;
                        }
                    }
                }
                else if ( mouseEventArgs.Button == MouseButtons.Right )
                {
                    if ( clickedId > 0 && mTileManager.SelectedSegmentId > 0 && clickedId != mTileManager.SelectedSegmentId)
                    {
                        //
                        // Perform the merge
                        //
                        switch ( mTileManager.CurrentMergeMode )
                        {
                            case MergeMode.Fill2D:
                                mTileManager.Internal.ReplaceSegmentationLabelCurrentSlice( clickedId, mTileManager.SelectedSegmentId, mTileManager.TiledDatasetView, p );
                                break;
                            case MergeMode.Fill3D:
                                mTileManager.Internal.ReplaceSegmentationLabelCurrentConnectedComponent( clickedId, mTileManager.SelectedSegmentId, mTileManager.TiledDatasetView, p );
                                break;
                            default:
                                mTileManager.Internal.ReplaceSegmentationLabel( clickedId, mTileManager.SelectedSegmentId );
                                break;
                        }
                        mTileManager.ChangesMade = true;
                    }
                }
            }
        }

        public override void OnMouseMove(MouseEventArgs mouseEventArgs, int width, int height)
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.TiledDatasetLoaded && mTileManager.SegmentationLoaded )
            {
                mCurrentlyHandlingMouseOver = true;
                //Mouseover - update display to highlight segment under mouse
                //Get the id of the segment being clicked

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float)mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float)mouseEventArgs.Y / height ) * extentDataSpace.Y;

                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                mTileManager.MouseOverSegmentId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                mCurrentlyHandlingMouseOver = false;
            }
        }

    }
}
