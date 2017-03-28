\header {
  objetivo = "dominante secundaria simples"
}
global = {
  \time 3/4
  \key g \major
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        g4 e d d e fis g2.
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        b4 c b a cis d b2.
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        g4 g g fis e d d2.
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        g4 c, g d' a d g,2.
      }
    >>
  >>
}
