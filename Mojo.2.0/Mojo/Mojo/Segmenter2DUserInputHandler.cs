using System;
using System.Windows.Forms;
using System.Windows.Input;

namespace Mojo
{
    internal class Segmenter2DUserInputHandler : IUserInputHandler
    {
        private readonly Segmenter mSegmenter;

        public Segmenter2DUserInputHandler( Segmenter segmenter )
        {
            mSegmenter = segmenter;
        }

        public void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {
            switch ( keyEventArgs.Key )
            {
                case Key.F2:
                    mSegmenter.InitializeSegmentation2D();
                    break;

                case Key.F3:
                    mSegmenter.InitializeSegmentation3D();
                    break;

                case Key.C:
                    mSegmenter.InitializeCostMap();
                    break;

                case Key.D:
                    mSegmenter.Internal.DumpIntermediateData();
                    break;

                case Key.S:
                    mSegmenter.ToggleShowSegmentation();
                    break;

                case Key.OemPlus:
                    mSegmenter.IncrementMaxForegroundCostDelta();
                    break;

                case Key.OemMinus:
                    mSegmenter.DecrementMaxForegroundCostDelta();
                    break;

                case Key.Escape:
                    mSegmenter.CommitSegmentation();
                    break;

                case Key.Back:
                    mSegmenter.ClearSegmentationAndCostMap();
                    break;

                case Key.Delete:
                    mSegmenter.ClearSegmentation();
                    break;

                case Key.Left:
                    mSegmenter.DecrementCurrentSlice();
                    break;

                case Key.Right:
                    mSegmenter.IncrementCurrentSlice();
                    break;

                case Key.Up:
                    mSegmenter.IncrementCurrentTexture();
                    break;

                case Key.Down:
                    mSegmenter.DecrementCurrentTexture();
                    break;
            }
        }

        public void OnMouseDown( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mSegmenter.DatasetLoaded )
            {
                var x = (int)Math.Floor( ( (float)mouseEventArgs.X / width ) * mSegmenter.Internal.GetVolumeDescription().NumVoxelsX );
                var y = (int)Math.Floor( ( (float)mouseEventArgs.Y / height ) * mSegmenter.Internal.GetVolumeDescription().NumVoxelsY );

                switch ( mSegmenter.CurrentSegmenterToolMode )
                {
                    case SegmenterToolMode.Adjust:
                        mSegmenter.BeginScribble( x, y );

                        if ( Keyboard.IsKeyDown( Key.LeftAlt ) )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mSegmenter.SelectSegmentationLabelOrScribble( x, y, ConstraintType.Foreground, Constants.MAX_BRUSH_WIDTH );
                            }

                            if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mSegmenter.SelectSegmentationLabelOrScribble( x, y, ConstraintType.Background, Constants.MAX_BRUSH_WIDTH );
                            }
                        }
                        else
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mSegmenter.SelectSegmentationLabelOrScribble( x, y, ConstraintType.Foreground, Constants.MIN_BRUSH_WIDTH );
                            }

                            if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mSegmenter.SelectSegmentationLabelOrScribble( x, y, ConstraintType.Background, Constants.MIN_BRUSH_WIDTH );
                            }
                        }
                        break;

                    case SegmenterToolMode.Merge:
                        //This is handled by MouseClick
                        break;

                    case SegmenterToolMode.Split:
                        mSegmenter.BeginScribble( x, y );

                        if ( Keyboard.IsKeyDown( Key.LeftAlt ) )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mSegmenter.SelectSplitSegmentationLabelOrScribble( x, y, ConstraintType.Foreground, Constants.MAX_BRUSH_WIDTH );
                            }

                            if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mSegmenter.SelectSplitSegmentationLabelOrScribble( x, y, ConstraintType.Background, Constants.MAX_BRUSH_WIDTH );
                            }
                        }
                        else
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mSegmenter.SelectSplitSegmentationLabelOrScribble( x, y, ConstraintType.Foreground, Constants.MIN_BRUSH_WIDTH );
                            }

                            if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mSegmenter.SelectSplitSegmentationLabelOrScribble( x, y, ConstraintType.Background, Constants.MIN_BRUSH_WIDTH );
                            }
                        }
                        break;

                    default:
                        Release.Assert( false );
                        break;
                }
            }
        }

        public void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mSegmenter.DatasetLoaded )
            {
                switch ( mSegmenter.CurrentSegmenterToolMode )
                {
                    case SegmenterToolMode.Adjust:
                        mSegmenter.EndScribble();
                        break;

                    case SegmenterToolMode.Merge:
                        break;

                    case SegmenterToolMode.Split:
                        break;

                    default:
                        Release.Assert( false );
                        break;
                }
            }
        }

        public void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if (mSegmenter.DatasetLoaded)
            {
                var x = (int)Math.Floor(((float)mouseEventArgs.X / width) * mSegmenter.Internal.GetVolumeDescription().NumVoxelsX);
                var y = (int)Math.Floor(((float)mouseEventArgs.Y / height) * mSegmenter.Internal.GetVolumeDescription().NumVoxelsY);

                switch (mSegmenter.CurrentSegmenterToolMode)
                {
                    case SegmenterToolMode.Adjust:
                        break;

                    case SegmenterToolMode.Merge:
                        if (mouseEventArgs.Button == MouseButtons.Left)
                        {
                            mSegmenter.SelectMergeSourceSegmentationLabel(x, y);
                        }

                        if (mouseEventArgs.Button == MouseButtons.Right)
                        {
                            if (mSegmenter.MergeSourceSegmentationLabel != null)
                            {
                                mSegmenter.SelectMergeDestinationSegmentationLabel(x, y);
                            }
                        }
                        break;

                    case SegmenterToolMode.Split:
                        break;

                    default:
                        Release.Assert( false );
                        break;
                }
            }
        }

        public void OnMouseMove( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mSegmenter.DatasetLoaded )
            {
                var x = (int)Math.Floor( ( (float)mouseEventArgs.X / width ) * mSegmenter.Internal.GetVolumeDescription().NumVoxelsX );
                var y = (int)Math.Floor( ( (float)mouseEventArgs.Y / height ) * mSegmenter.Internal.GetVolumeDescription().NumVoxelsY );

                switch ( mSegmenter.CurrentSegmenterToolMode )
                {
                    case SegmenterToolMode.Adjust:
                        if ( Keyboard.IsKeyDown( Key.LeftAlt ) )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mSegmenter.Scribble( x, y, ConstraintType.Foreground, Constants.MAX_BRUSH_WIDTH );
                            }

                            if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mSegmenter.Scribble( x, y, ConstraintType.Background, Constants.MAX_BRUSH_WIDTH );
                            }
                        }
                        else
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mSegmenter.Scribble( x, y, ConstraintType.Foreground, Constants.MIN_BRUSH_WIDTH );
                            }

                            if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mSegmenter.Scribble( x, y, ConstraintType.Background, Constants.MIN_BRUSH_WIDTH );
                            }
                        }
                        break;

                    case SegmenterToolMode.Merge:
                        break;

                    case SegmenterToolMode.Split:
                        if ( mSegmenter.SplitSegmentationLabel != null )
                        {
                            if ( Keyboard.IsKeyDown( Key.LeftAlt ) )
                            {
                                if ( mouseEventArgs.Button == MouseButtons.Left )
                                {
                                    mSegmenter.Scribble( x, y, ConstraintType.Foreground, Constants.MAX_BRUSH_WIDTH );
                                }

                                if ( mouseEventArgs.Button == MouseButtons.Right )
                                {
                                    mSegmenter.Scribble( x, y, ConstraintType.Background, Constants.MAX_BRUSH_WIDTH );
                                }
                            }
                            else
                            {
                                if ( mouseEventArgs.Button == MouseButtons.Left )
                                {
                                    mSegmenter.Scribble( x, y, ConstraintType.Foreground, Constants.MIN_BRUSH_WIDTH );
                                }

                                if ( mouseEventArgs.Button == MouseButtons.Right )
                                {
                                    mSegmenter.Scribble( x, y, ConstraintType.Background, Constants.MIN_BRUSH_WIDTH );
                                }
                            }
                        }
                        break;

                    default:
                        Release.Assert( false );
                        break;
                }
            }
        }

        public void OnMouseWheel( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mouseEventArgs.Delta > 0 )
            {
                mSegmenter.IncrementCurrentSlice();
            }
            else
            if ( mouseEventArgs.Delta < 0 )
            {
                mSegmenter.DecrementCurrentSlice();
            }
        }

        public void SetSize( int oldWidth, int oldHeight, int newWidth, int newHeight )
        {
        }
    }
}
