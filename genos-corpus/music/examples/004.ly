\header {
  objetivo = "tonica dominante tonica com inversoes"
}
global = {
  \time 3/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        c d e
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        c b c
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
        e d c
      }
    >>
  >>
}
