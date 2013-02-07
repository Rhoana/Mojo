using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Data;
using System.ComponentModel;

namespace Mojo.Wpf.ViewModel
{
    public class PagedSegmentListView : CollectionView
    {

        public IList mInnerList;

        private int mItemsPerPage;

        private int mCurrentPage = 0;

        private int mCurrentItemIndex = 0;

        public override object CurrentItem
        {
            get
            {
                return GetItemAt( mCurrentItemIndex );
            }
        }

        public PagedSegmentListView( IList innerList, int itemsPerPage )
            : base( innerList )
        {
            mInnerList = innerList;
            mItemsPerPage = itemsPerPage;
        }

        public void SetItemsPerPage( int nItems )
        {
            var firstItemIndex = mCurrentPage * mItemsPerPage;
            mItemsPerPage = nItems;
            mCurrentPage = firstItemIndex / mItemsPerPage;
        }

        public override int Count
        {
            get { 
                if ( CurrentPage == PageCount - 1 )
                    return mInnerList.Count - ( PageCount - 1 ) * mItemsPerPage;
                else
                    return mItemsPerPage;
                }
        }

        public int CurrentPage
        {
            get { return mCurrentPage; }
            set
            {
                mCurrentPage = value;
                OnPropertyChanged( new PropertyChangedEventArgs( "SegmentListCurrentPage" ) );
            }
        }

        public int ItemsPerPage { get { return mItemsPerPage; } }

        public int PageCount
        {
            get
            {
                return ( mInnerList.Count + mItemsPerPage - 1 ) / mItemsPerPage;
            }
        }

        public override object GetItemAt( int index )
        {
            var listIndex = mItemsPerPage * mCurrentPage + index;
            return (listIndex >= mInnerList.Count) ? null : mInnerList[listIndex];
        }

        public override bool MoveCurrentToPosition( int position )
        {
            mCurrentItemIndex = position;
            return true;
        }

        public void MoveToNextPage()
        {
            if ( mCurrentPage < PageCount - 1 )
            {
                CurrentPage += 1;
            }
            Refresh();
        }

        public void MoveToPreviousPage()
        {
            if ( mCurrentPage > 0 )
            {
                CurrentPage -= 1;
            }
            Refresh();
        }

        public void MoveToFirstPage()
        {
            CurrentPage = 0;
            Refresh();
        }

        public void MoveToLastPage()
        {
            CurrentPage = PageCount - 1;
            Refresh();
        }


    }

}
