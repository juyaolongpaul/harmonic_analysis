\header {
  objetivo = "napolitano e primeira inversao"
}
global = {
  \time 4/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        c des b c
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        g aes g g
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        e des f e
      }
    >>
    \new Staff <<
      \global
      \relative c {
        e f g c,
      }
    >>
  >>
}
