\header {
  objetivo = "acorde aumentado"
}
global = {
  \key c \minor
  \time 4/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c''' {
        g4 f ees d
        c1
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        g4 aes g g
        ees1
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        c4 c b b
        g1
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        ees4 f g g
        c,1
      }
    >>
  >>
}