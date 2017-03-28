\header {
  objetivo = "acorde com sétima maior"
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
        \partial 4
        c4
        c b c
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        g4
        a g g
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        e4
        e d e
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        e4
        f g c,
      }
    >>
  >>
}