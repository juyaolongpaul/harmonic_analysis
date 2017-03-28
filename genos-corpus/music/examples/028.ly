\header {
  objetivo = "acorde aumentado"
  fonte = "morris:figured. p.43"
}
global = {
  \key c \major
  \time 4/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        a4 gis a2
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        f4 e e2
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        c4 c a2
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        f4 c cis2
      }
    >>
  >>
}