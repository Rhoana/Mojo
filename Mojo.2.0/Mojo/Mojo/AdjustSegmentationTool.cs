using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    class AdjustSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        private bool mCurrentlyDrawing = false;
        private bool mCurrentlyMovingZ = false;

        public AdjustSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
            mEngine = engine;
        }

        public override void Select()
        {
            if ( mTileManager.SelectedSegmentId != 0 )
            {
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForAdjust( p );
            }
        }

        public override void SelectSegment( uint segmentId )
        {
            if ( mTileManager.SelectedSegmentId != segmentId )
            {
                mTileManager.SelectedSegmentId = segmentId;
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForAdjust( p );
            }
        }

        public override void MoveZ()
        {
            if ( mTileManager.SelectedSegmentId != 0 && !mCurrentlyMovingZ )
            {
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForAdjust( p );
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {
            var p = new Vector2( 0.5f, 0.5f );

            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.W:
                    mCurrentlyMovingZ = true;
                    break;
                case System.Windows.Input.Key.S:
                    mCurrentlyMovingZ = true;
                    break;
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
                    {
                        mTileManager.UndoChange();
                        mTileManager.PrepForAdjust( p );
                    }
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.RedoChange();
                        mTileManager.PrepForAdjust( p );
                    }
                    break;
                case System.Windows.Input.Key.N:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.SelectNewId();
                        mTileManager.PrepForAdjust( p );
                    }
                    break;
                case System.Windows.Input.Key.Tab:
                    mTileManager.CommmitAdjustChange();
                    keyEventArgs.Handled = true;
                    break;

                case System.Windows.Input.Key.Escape:
                    mTileManager.CancelAdjustChange();
                    mTileManager.SelectedSegmentId = 0;
                    break;

                case System.Windows.Input.Key.OemComma:
                case System.Windows.Input.Key.OemMinus:
                case System.Windows.Input.Key.Subtract:
                    mTileManager.DecreaseBrushSize();
                    break;
                case System.Windows.Input.Key.OemPeriod:
                case System.Windows.Input.Key.OemPlus:
                case System.Windows.Input.Key.Add:
                    mTileManager.IncreaseBrushSize();
                    break;

            }

            base.OnKeyDown( keyEventArgs, width, height );

        }

        public override void OnKeyUp(System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height)
        {
            base.OnKeyUp(keyEventArgs, width, height);
            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.W:
                    mCurrentlyMovingZ = false;
                    MoveZ();
                    break;
                case System.Windows.Input.Key.S:
                    mCurrentlyMovingZ = false;
                    MoveZ();
                    break;
            }
        }
        
        public override void OnMouseDown(System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height)
        {
            base.OnMouseDown( mouseEventArgs, width, height );
            if ( mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress && mTileManager.SelectedSegmentId != 0 )
            {
                //
                // Draw or erase here
                //

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                mCurrentlyDrawing = true;

                if ( mouseEventArgs.Button == MouseButtons.Left )
                {
                    mTileManager.DrawRegionA( p );
                    mEngine.QuickRender();
                }
                else if ( mouseEventArgs.Button == MouseButtons.Right )
                {
                    mTileManager.DrawRegionB( p );
                    mEngine.QuickRender();
                }
            }
        }

        public override void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseUp( mouseEventArgs, width, height );
            mCurrentlyDrawing = false;
        }

        public override void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress && mTileManager.SelectedSegmentId == 0 )
            {
                //Get the id of the segment being clicked

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                var clickedId = mTileManager.GetSegmentationLabelId( p );

                if ( clickedId > 0 && mouseEventArgs.Button != MouseButtons.Middle )
                {
                    //
                    // Select this segment
                    //
                    mTileManager.SelectedSegmentId = clickedId;
                    mTileManager.PrepForAdjust( p );
                }
            }
        }

        public override void OnMouseMove( MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.SourceImagesLoaded && mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress )
            {
                mCurrentlyHandlingMouseOver = true;

                //
                // Mouseover - update display to highlight segment or area under mouse
                // Get the id of the segment under the mouse
                //

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                mTileManager.MouseOver( p );

                if ( mTileManager.SelectedSegmentId != 0 )
                {
                    if ( mCurrentlyDrawing )
                    {
                        if ( mouseEventArgs.Button == MouseButtons.Left )
                        {
                            mTileManager.DrawRegionA( p );
                            mEngine.QuickRender();
                        }
                        else if ( mouseEventArgs.Button == MouseButtons.Right )
                        {
                            mTileManager.DrawRegionB( p );
                            mEngine.QuickRender();
                        }
                    }
                }

                mCurrentlyHandlingMouseOver = false;

            }
        }

    }
}
