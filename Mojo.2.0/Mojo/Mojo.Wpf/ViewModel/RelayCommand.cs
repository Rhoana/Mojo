using System;
using System.Windows.Input;

namespace Mojo.Wpf.ViewModel
{
    /// <summary>
    /// A command whose sole purpose is to 
    /// relay its functionality to other
    /// objects by invoking delegates. The
    /// default return value for the CanExecute
    /// method is 'true'.
    /// </summary>
    public class RelayCommand : ICommand
    {
        readonly Action< object > mExecute;
        readonly Predicate< object > mCanExecute;        

        /// <summary>
        /// Creates a new command that can always execute.
        /// </summary>
        /// <param name="execute">The execution logic.</param>
        public RelayCommand( Action< object > execute ) : this( execute, null )
        {
        }

        /// <summary>
        /// Creates a new command.
        /// </summary>
        /// <param name="execute">The execution logic.</param>
        /// <param name="canExecute">The execution status logic.</param>
        public RelayCommand( Action<object> execute, Predicate<object> canExecute )
        {
            if ( execute == null )
            {
                throw new ArgumentNullException( "execute" );                
            }

            mExecute = execute;
            mCanExecute = canExecute;           
        }

        public bool CanExecute( object parameter )
        {
            return mCanExecute == null || mCanExecute( parameter );
        }

        private event EventHandler CanExecuteChangedInternal;

        public event EventHandler CanExecuteChanged
        {
            add
            {
                CommandManager.RequerySuggested += value;
                CanExecuteChangedInternal += value;
            }
            remove
            {
                CommandManager.RequerySuggested -= value;
                CanExecuteChangedInternal -= value;
            }
        }

        public void Execute( object parameter )
        {
            mExecute( parameter );
        }

        public void RaiseCanExecuteChanged()
        {
            var handler = CanExecuteChangedInternal;

            if ( handler != null )
            {
                handler( this, new EventArgs() );
            }
        }
    }
}