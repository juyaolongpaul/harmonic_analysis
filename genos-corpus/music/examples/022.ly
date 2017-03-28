\header {
  objetivo = "pequena tonicizacao"
}
global = {
  \key e \major
  \time 4/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        e4 e fis e
        e e dis e
      }
    >>
    \new Staff <<
      \global
      \relative c'' {
        gis4 gis fis gis
        gis a a gis
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        \clef bass
        b4 cis dis dis
        cis cis b b
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        e4 a b e,
        cis fis b, e
      }
    >>
  >>
}