\version "2.12.0"
\new PianoStaff <<
  \new Staff {
    \time 2/2
    \relative c'' {
      c1(\p^\markup {"Vl.I"}
      d 
      f 
      e)
    }
  }
  \new Staff  {
    \time 2/2
    \relative c' {
      e8^\markup {"Vl.II"} c' e, c' e, c' e, c'
      f, c' f, c' f, c' f, c'
      g b g b g b g b
      a c a c a c a c  
    }
  }
>>
