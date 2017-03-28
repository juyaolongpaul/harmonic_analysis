\header {
  objetivo = "dominante secundaria com inversoes"
}
global = {
  \time 3/4
  \key g \minor
}
\score {
  <<
    \new Staff <<
      \global
      \relative c' {
        d4 d d ees g fis g2.
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        bes4 a g c d d d2.
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        g4 fis g g g a bes2.
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        g4 a b c bes a g2.
      }
    >>
  >>
}
