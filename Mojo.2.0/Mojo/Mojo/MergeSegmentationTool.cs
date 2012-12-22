using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    public class MergeSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private int newId = 0;

        public MergeSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {
            base.OnKeyDown( keyEventArgs, width, height );

            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.A:
                    mTileManager.ToggleShowSegmentation();
                    break;
                case System.Windows.Input.Key.E:
                    mTileManager.SegmentationVisibilityRatio = System.Math.Min( mTileManager.SegmentationVisibilityRatio + 0.1f, 1.0f );
                    break;
                case System.Windows.Input.Key.D:
                    mTileManager.SegmentationVisibilityRatio = System.Math.Max( mTileManager.SegmentationVisibilityRatio - 0.1f, 0f );
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
                        if ( newId == clickedId )
                        {
                            //Unselect this segment
                            //newId = 0;
                            //mTileManager.SelectedSegmentId = 0;
                        }
                        else
                        {
                            //Select this segment
                            newId = clickedId;
                            mTileManager.SelectedSegmentId = clickedId;
                        }
                    }
                }
                else if ( mouseEventArgs.Button == MouseButtons.Right )
                {
                    //Update this segment
                    if ( clickedId > 0 && newId > 0 )
                    {
                        if ( mTileManager.ConstrainSegmentationMergeToCurrentSlice )
                        {
                            mTileManager.Internal.ReplaceSegmentationLabelCurrentSlice( clickedId, newId, mTileManager.TiledDatasetView, p );
                        }
                        else if ( mTileManager.ConstrainSegmentationMergeToConnectedComponent )
                        {
                            mTileManager.Internal.ReplaceSegmentationLabelCurrentConnectedComponent( clickedId, newId, mTileManager.TiledDatasetView, p );
                        }
                        else
                        {
                            mTileManager.Internal.ReplaceSegmentationLabel( clickedId, newId );
                        }
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

                int segmentId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                if ( segmentId > 0 )
                {
                    mTileManager.MouseOverSegmentId = segmentId;
                }
                else
                {
                    mTileManager.MouseOverSegmentId = 0;
                }

                mCurrentlyHandlingMouseOver = false;
            }
        }

    }
}
