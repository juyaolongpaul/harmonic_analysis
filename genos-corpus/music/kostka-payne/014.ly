\version "2.12.0"
\header {
  title = "Sonata Op. 26, I"
  composer = "Beethoven"
}

\new PianoStaff <<
  \new Staff {
    \time 3/8
    \key aes \major
    \relative c' {
      \partial 8 ees8\p
      <<
        { 
          ees8( aes) aes-.
          aes16( g) g16.( f32 g16. ees32)
          aes4( bes8)
          des16.(\sf\> c32) bes8-. \grace c8 bes32(\p a bes c)
        }
        \\
        { 
          c,4.
          <des ees>4.
          r8 ees <bes ees>
          <c ees aes> <bes ees g> s8
        }
      >>
      <aes'! des>4( <aes c>8
      <aes bes f'>8 <g bes> <aes ees'>)
      << 
        {
          <f aes>(\p <fes bes des> <ees aes c>
          <fis a> <g bes>16) r ees16. ees32
          ees16( aes!) aes-. aes-. aes-. aes-.
          aes( g) g16.( f32 g16. ees32)
          \break
          aes4( bes8)
          des16.(\sf\> c32) bes8-. \grace c8 bes32(\p a bes c)				
        } 
        \\
        {
          s4.
          ees,4 s8
          c4.
          <des ees>4.
          s8 ees <bes ees>
          <c ees aes> <bes ees g> s8			
        } 
      >>
      <aes'! des>4 <aes c>8
      <aes bes f'>8( <g bes> <aes ees'>)
      \grace bes8 aes32( g aes bes) <ees, aes c>8( <des g bes>
      <c aes'>8) r8
    }
  }
  \new Staff  {
    \time 3/8
    \key aes \major
    \clef bass
    \relative c{
      \partial 8 ees8
      << 
        {
          ees( aes) aes-. 
          aes16( g) g16.( f32 g16. ees32)
          aes4 ees8
        } 
        \\
        {
          aes,4.
          <bes ees>4.
          r8 c g
        } 
      >>	
      <aes ees'> <ees ees'> r8
      f'( f' ees d des c)
      f,( g aes)
      << 
        {
          s4 ees16. ees32
          ees16( aes) aes-. aes-. aes-. aes-.
          aes( g) g16.( f32 g16. ees32)
          aes4 ees8 
        } 
        \\
        {
          ees4 r8
          aes,4.
          <bes ees>4.
          r8 c g
        } 
      >>
      <aes ees'> <ees ees'> r8
      f'( f' ees d des c)
      r8 ees,( ees,-.)
      aes-. aes,
    }
  }
>>