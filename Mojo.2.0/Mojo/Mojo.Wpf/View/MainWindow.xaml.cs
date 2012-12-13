using System;
using System.Windows.Controls;

namespace Mojo.Wpf.View
{
    public partial class MainWindow : IDisposable
    {
        public MainWindow()
        {
            InitializeComponent();

            SegmentationLabelesListBox.SelectionChanged += SegmentationLabelListBoxSelectionChangedHandler;
        }

        public void Dispose()
        {
            SegmentationLabelesListBox.SelectionChanged -= SegmentationLabelListBoxSelectionChangedHandler;
        }

        private void SegmentationLabelListBoxSelectionChangedHandler( object sender, SelectionChangedEventArgs e )
        {
            SegmentationLabelesListBox.ScrollIntoView( SegmentationLabelesListBox.SelectedItem );
        }
    }
}
