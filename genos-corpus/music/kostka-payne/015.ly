\version "2.12.0"
\new PianoStaff <<
  \new Staff {
    \key f \major
    \relative c'' {
      <<
        { 
          f2 f4 e? \bar "||"
          f2 e?  \bar "||"
          f2 e?  \bar "||"
          f2 e?  \bar "||"
          f,2 e?  \bar "||"
          d2 c  \bar "||"
        }
        \\
        { 
          bes'2 a4 g
          g2 g
          <g bes>2 <g bes>
          <g d'>2 <g c>
          bes,2 bes
          bes2 bes
        }
      >>
    }
  }
  \new Staff  {
    \key f \major
    \clef bass
    \relative c'{
      <<
        { 
          d2 c
          d2 c
          s2 s2
          s2 s2
          g2 g
          g2 g
        }
        \\
        { 
          g2 c
          bes2 c
          g g
          bes bes
          d, c
          f e?
        }
      >>
    }
  }
>>