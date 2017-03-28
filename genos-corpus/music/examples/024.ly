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
        g4 gis a
        d dis e
        g, gis a
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        e4 e f
        g g g
        e e e
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        c4 c c
        b b c
        c c c
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        c4 c f
        g g c,
        c c c
      }
    >>
  >>
}