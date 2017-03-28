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
        d4 dis e2
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        b4 b c2
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        \clef bass
        g4 g g2
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        g4 f e2
      }
    >>
  >>
}