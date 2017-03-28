\version "2.12.0"

\new PianoStaff <<
  \new Staff {
    \set Staff.instrumentName = \markup {\column {"soprano" \line {"alto"}}}
    \key ees \major
    \partial 4
    \relative c'' {
      << 
        {
          ees
          d c bes aes8 g
          c4 c bes\fermata 
        } 
        \\
        {
          bes
          bes g g f8 ees
          ees4 aes g 
        }
      >>
    }
  }
  \new Staff {
    \set Staff.instrumentName = \markup {\column {"tenor" \line {"bass"}}}
    \clef bass
    \key ees \major
    \partial 4
    \relative c'' {
      <<
        {
          g
          f ees d des
          c8 bes c d! ees4 
        } 
        \\
        {  
          ees,
          bes' c g bes
          aes aes, ees'\fermata 
        }
      >>
    }
  }
>>