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
        g4 gis a2
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        c4 c c2
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        c4 bes a2
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        e4 e f2
      }
    >>
  >>
}