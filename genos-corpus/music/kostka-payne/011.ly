\version "2.12.0"
\new PianoStaff <<
  \new Staff {
    \relative c'{
      \time 4/4
      \key a \major
      fis4 gis a fis 
      eis fis b gis 
      a2 r2
      \bar "||"
    }
  }
  \new Staff {
    \relative c{
      \time 4/4
      \key a \major
      \clef bass
      fis4 e a, b 
      cis d b cis 
      fis,2 r2
    }
  }
>>