\header {
  objetivo = "pequena tonicizacao"
}
global = {
  \key c \minor
  \time 3/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        c4 b c
        c c cis
        d d d
        e d d
        c b c
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        g4 g g
        aes g g
        fis g fis
        g fis f
        ees f ees
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        ees4 d ees
        f ees e
        d bes a
        bes a b
        c d c
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        c4 g c
        f c a
        d g d
        cis d g,
        c g c
      }
    >>
  >>
}