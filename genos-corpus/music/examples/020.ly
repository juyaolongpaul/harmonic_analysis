\header {
  objetivo = "pequena tonicizacao"
}
global = {
  \key d \major
  \time 4/4
}
\score {
  <<
    \new Staff <<
      \global
      \relative c'' {
        a4 g fis eis
        fis gis a b
        b d cis d
      }
    >>
    \new Staff <<
      \global
      \relative c' {
        d4 e d b
        a b a d
        e d e d
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        fis4 e fis gis
        fis eis fis fis
        g fis g fis
      }
    >>
    \new Staff <<
      \global
      \relative c {
        \clef bass
        d4 a d cis
        fis cis fis b,
        g a2 d4
      }
    >>
  >>
}