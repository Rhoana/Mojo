using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    public class MergeSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        public MergeSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
            mEngine = engine;
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
                        mTileManager.UndoChange();
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                        mTileManager.RedoChange();
                    break;
                case System.Windows.Input.Key.N:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                        mTileManager.SelectNewId();
                    break;
                case System.Windows.Input.Key.Escape:
                    //Unselect this segment
                    mTileManager.SelectedSegmentId = 0;
                    break;

            }

            base.OnKeyDown( keyEventArgs, width, height );

        }

        public override void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress )
            {
                //Get the id of the segment being clicked

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                var clickedId = mTileManager.GetSegmentationLabelId( p );

                if ( mouseEventArgs.Button == MouseButtons.Left )
                {
                    //Select this segment
                    if ( clickedId > 0 )
                    {
                        if ( clickedId != mTileManager.SelectedSegmentId )
                        {
                            //Select this segment
                            mTileManager.SelectedSegmentId = clickedId;
                            mEngine.UpdateExternalViewerLocation( mTileManager.GetPointDataSpace( p ) );
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
                                mTileManager.ReplaceSegmentationLabelCurrentSlice( clickedId, p );
                                break;
                            case MergeMode.Fill3D:
                                mTileManager.ReplaceSegmentationLabelCurrentConnectedComponent( clickedId, p );
                                break;
                            default:
                                mTileManager.RemapSegmentLabel( clickedId );
                                break;
                        }
                    }
                }
            }
        }

        public override void OnMouseMove(MouseEventArgs mouseEventArgs, int width, int height)
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.TiledDatasetLoaded && mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress )
            {
                mCurrentlyHandlingMouseOver = true;
                //Mouseover - update display to highlight segment under mouse
                //Get the id of the segment being clicked

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                mTileManager.MouseOverSegmentId = mTileManager.GetSegmentationLabelId( p );

                mCurrentlyHandlingMouseOver = false;
            }
        }

    }
}
