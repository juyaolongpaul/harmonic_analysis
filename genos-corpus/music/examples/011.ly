\header {
  objetivo = "acorde de sexta alema"
}
\score {
  <<
    \new Staff {
      \relative c'' {
        a b c b
        a g f e
        d dis e e
        e1
        \bar "|."
      }
    }
    \new Staff {
      \relative c' {
        e4 e e e
        e e d c
        d4 c c b
        c1
        \bar "|."
      }
    }
    \new Staff {
      \relative c' {
        \clef bass
        c4 b a gis
        a a a a
        a4 a a gis
        a1
        \bar "|."
      }
    }
    \new Staff {
      \relative c {
        \clef bass
        a4 gis a b
        c cis d e
        f f e e
        a,1
        \bar "|."
      }
    }
  >>
  \midi {}
  \layout {}
}