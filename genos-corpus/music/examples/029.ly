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
        gis2 a
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        e2 f
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        bes2 a
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        c2 f
      }
    >>
  >>
}