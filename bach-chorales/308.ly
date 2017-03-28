\version "2.10.33"

\header {
  title = "308 - Ach Gott, wie manches Herzeleid"
  composer = "J. S. Bach"
}

global = {
  \time 4/4
  \key a \major
}

soprano = \relative c'' {
  \partial 4 a4
  a a fis8 gis a4
  b8 a gis4 fis\fermata fis
  gis a b e,8 fis
  gis a fis4 e\fermata b'
  cis cis a8 b cis4
  d cis b\fermata a
  b cis b a
  a gis a\fermata
}

alto = \relative c' {
  \partial 4 cis8 d 
  e4 a, d cis
  b e dis dis
  e fis fis8 dis e4~e8 fis dis4 b gis'
  a cis, fis8 gis ais4
  b a gis fis
  e8 d cis dis e d cis4
  fis e e
}

tenor = \relative c' {
  \partial 4 a8 b
  cis d e4 a, a8 gis
  fis4 gis8 a b4 b
  b dis gis,8 a b cis
  b4 b8 a gis4 e'
  e8 fis gis4 cis,8 d e4
  fis8 gis e fis e4 cis
  b e,8 fis gis4 a
  b8 cis d4 cis
}

baixo = \relative c {
  \partial 4 a'4
  a,8 b cis4 d8 e fis e
  dis4 e b\fermata b'8 a
  gis a gis fis e fis gis a
  b4 b, e\fermata e
  a8 gis fis eis fis e d cis
  b4 cis8 d e4\fermata fis
  gis a e fis8 e
  dis4 e a,\fermata
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