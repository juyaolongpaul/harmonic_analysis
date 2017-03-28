\version "2.12.0"
\header {
  title = "Ruf zur Maria, Op 22"
  composer = "Brahms"
}
\new ChoirStaff {
  \set Score.currentBarNumber = #28
  \bar ""
  <<
    \new Staff 
    \relative c''{
      \time 6/4
      \key bes \major
      r2. c2^\p cis4
      d2 bes4 d2( c!4)
      \partial 4*5 bes2.^\> ~ bes2\!
      \bar "|."
    }
    \addlyrics { 
      bitt für uns, Ma- ri- a! 
    }
    
    \new Staff 
    \relative c''{
      \key bes \major
      r2. g2^\p ges4
      f2 d4 a'!2.
      f4^\> ( g ees f2\! )
    }
    
    \new Staff 
    \relative c'{
      \key bes \major
      \clef "G_8"
      r2. bes2^\p bes4 bes( d) f f2( ees4)
      d^\> ( ees c d2\!)
    }
    \addlyrics { 
      bitt für uns, Ma- ri- a! 
    }

    \new Staff 
    \relative c{
      \clef bass \key bes \major
      r2. ees2^\p e4
      f2 f4 f2.
      bes2.^\> ~ bes2\!
    }
  >>
}