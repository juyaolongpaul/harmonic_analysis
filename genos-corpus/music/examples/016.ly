\header {
  objetivo = "tonica diminuto tonica"
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
        g f g
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
