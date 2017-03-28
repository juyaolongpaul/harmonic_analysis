\header {
  objetivo = "acorde aumentado"
}
global = {
  \key c \major
  \time 3/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        e4 dis e
        e dis e
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        c4 b c
        g g g
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        \clef bass
        g4 g g
        g g g
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        c4 g c
        c b c
      }
    >>
  >>
}