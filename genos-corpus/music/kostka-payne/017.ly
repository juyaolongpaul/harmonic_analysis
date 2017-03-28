\version "2.12.0"
\header {
  title = "Symphony 9"
  composer = "Dvorák"
}

\new ChoirStaff <<
  \new Staff { 
    \relative c' { 
      \time 4/8  
      \clef alto  
      \key g \major
      g4(\pp^\markup {"Vla"} fis8 f)
      e4( ees8 d)
      cis8( c)
      << 
        {
          g'16 r fis r g r
        } 
        \\ 
        {
          e16 s e s e s
        }
      >>
    }
  }
  \new Staff { 
    \relative c' { 
      \clef tenor 
      \time 4/8
      \key g \major
      e8(\pp^\markup {"Vc"} ~ e32 d16 e32 b4)
      c8( ~ c32 b16 c32 g4)
      a8( ~ a32 g16 a32) b16 r c r
      b r
    } 
  }
  \new Staff { 
    \relative c' { 
      \time 4/8 
      \clef bass 
      \key g \major
      e4(\pp^\markup {"D.B."} d)
      c( b)
      a e16 r a r
      e r
    } 
  }
>>