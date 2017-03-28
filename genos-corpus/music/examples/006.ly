\header {
  objetivo = "tonica dominante com 7 tonica com inversoes"
}
global = {
  \time 3/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        c d c
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        g g g
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        e f e
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        c b c
      }
    >>
  >>
}
