\version "2.10.33"

\header {
  title = "236 - O Jesu, du mein Bräutigam"
  composer = "J. S. Bach"
}

global = {
  \time 4/4
  \key g \major
}

soprano = \relative c'' {
  \partial 4 b
  b a b g
  fis8 g16 a fis4 e4\fermata e8 fis
  g4 a b a
  d cis b\fermata b
  e fis e b
  cis b a\fermata a8 b
  cis4 dis e d
  cis8 d16 e cis4 b\fermata
}

alto = \relative c' {
  \partial 4 dis4
  e e fis e
  e dis b b
  e d d d8 e
  fis4 fis8. e16 dis4 e
  e d e eis
  fis4. eis8 fis4 fis8 gis
  a4 a b8 cis~cis b
  b ais16 gis ais4 fis
}

tenor = \relative c' {
  \partial 4 fis,
  g a8 g fis4 b
  c b8 a g4 g8 a
  b4 a g fis
  b~b8 ais fis4 g
  a a8 b cis4 d
  cis gis8 cis cis4 d
  e fis e8 cis d fis
  g cis, fis e dis4
}

baixo = \relative c {
  \partial 4 b4
  e c' dis, e
  a, b e,\fermata e
  e' fis g d8 cis
  b4 fis' b,\fermata e8 d
  cis4 d a' gis
  a8 b cis4 fis,\fermata d
  a' gis8 fis gis ais b4
  e, fis b,\fermata
}
\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}