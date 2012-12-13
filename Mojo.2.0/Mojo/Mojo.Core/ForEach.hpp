#pragma once

class IForEachContainer {};

template < typename T >
class ForEachContainer : public IForEachContainer
{
public:
    ForEachContainer( const T& container );

    bool Finished() const;
    inline void Iterate();

    bool FinishedMiddle() const;
    void IterateMiddle();

    bool FinishedInner() const;
    void IterateInner();

    typename T::const_iterator GetIterator();

private:
    int                                mMiddleLoopCount;
    int                                mInnerLoopCount;
    mutable typename T::const_iterator mIterator;
    mutable typename T::const_iterator mEnd;
};

// helper methods for for each macro
template < typename T > inline ForEachContainer< T >  CreateForEachContainer( const T& t );
template < typename T > inline ForEachContainer< T >* GetForEachContainer   ( IForEachContainer* base, T* );
template < typename T > inline T*                     TypeSafeDummy         ( const T& );


template < typename T > ForEachContainer< T >::ForEachContainer( const T& container ) :
    mInnerLoopCount ( 0 ),
    mMiddleLoopCount( 0 ),
    mIterator       ( container.begin() ),
    mEnd            ( container.end()   )
{
};

template < typename T > bool ForEachContainer< T >::Finished() const
{
    return mIterator == mEnd;
};

template < typename T > void ForEachContainer< T >::Iterate()
{
    mIterator++;
    mMiddleLoopCount = 0;
    mInnerLoopCount  = 0;
};

template < typename T > bool ForEachContainer< T >::FinishedMiddle() const
{
    return mMiddleLoopCount > 0;
};

template < typename T > void ForEachContainer< T >::IterateMiddle()
{
    mMiddleLoopCount++;
    mInnerLoopCount = 0;
};

template < typename T > bool ForEachContainer< T >::FinishedInner() const
{
    return mInnerLoopCount > 0;
};

template < typename T > void ForEachContainer< T >::IterateInner()
{
    mInnerLoopCount++;
};

template < typename T > typename T::const_iterator ForEachContainer< T >::GetIterator()
{
    return mIterator;
};


// used to create an indirection so the for loop doesn't have to know the container type
template < typename T > inline ForEachContainer< T > CreateForEachContainer( const T& t )
{
    return ForEachContainer< T >( t );
}

// gets a pointer to the derived container from the base container
template < typename T > inline ForEachContainer< T >* GetForEachContainer( const IForEachContainer* base, T* )
{
    return static_cast< ForEachContainer< T >* >( const_cast<IForEachContainer*>(base) );
}

// always returns a pointer of type T that equals null
template < typename T > inline T* TypeSafeDummy( const T& )
{
    return NULL;
}

#define MOJO_PROTECT_COMMAS(...) __VA_ARGS__

#define MOJO_FOR_EACH(loopVariable,container)                                                                        \
    for (                                                                                                            \
        const IForEachContainer& forEachContainer = CreateForEachContainer( container );                             \
        !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Finished() );                       \
         ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Iterate() ) )                       \
        for (                                                                                                        \
            loopVariable = *( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator() ); \
            !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedInner() );              \
             ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateInner() ) )

#define MOJO_FOR_EACH_KEY(loopVariable,container)                                                                          \
    for (                                                                                                                  \
        const IForEachContainer& forEachContainer = CreateForEachContainer( container );                                   \
        !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Finished() );                             \
         ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Iterate() ) )                             \
        for (                                                                                                              \
            loopVariable = ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator()->first ); \
            !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedInner() );                    \
             ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateInner() ) )

#define MOJO_FOR_EACH_VALUE(loopVariable,container)                                                                         \
    for (                                                                                                                   \
        const IForEachContainer& forEachContainer = CreateForEachContainer( container );                                    \
        !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Finished() );                              \
         ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Iterate() ) )                              \
        for (                                                                                                               \
            loopVariable = ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator()->second ); \
            !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedInner() );                     \
             ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateInner() ) )


#define MOJO_FOR_EACH_KEY_VALUE(loopKey,loopValue,container)                                                                   \
    for (                                                                                                                      \
        const IForEachContainer& forEachContainer = CreateForEachContainer( container );                                       \
        !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Finished() );                                 \
         ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Iterate() ) )                                 \
        for ( loopKey = ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator()->first );        \
            !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedMiddle() );                       \
             ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateMiddle() ) )                       \
            for ( loopValue = ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator()->second ); \
                !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedInner() );                    \
                 ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateInner() ) )

