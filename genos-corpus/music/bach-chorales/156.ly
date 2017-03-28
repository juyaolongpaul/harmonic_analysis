\version "2.10.33"

\header {
  title = "156 - Ach Gott, wie manches Herzeleid"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 a4 fis8 gis8 a4 
  b8 a8 gis4 fis4 fis4 
  gis4 a4 b4 e,8 fis8 
  gis8 a8 fis4 e4 b'4 
  cis4 cis4 a8 b8 cis4   %5
  d4 cis4 b4 a4 
  b4 cis4 b4 a4 
  a4 gis4 a4 
}

alto = \relative c' {
  \partial 4 cis8  d8 
    e4 a,4 d4 cis4 
  b4 e4 dis4 dis4 
  e4 fis4 fis8 dis8 e4 
  e8 fis8 dis4 b4 gis'4 
  a4 cis,4 fis8 gis8 ais4   %5
  b4 e,8 fis8 gis4 fis4 
  e8 d8 cis8 dis8 e8 d8 cis4 
  fis4 e4 e4 
}

tenor = \relative c' {
  \partial 4 a8  b8 
    cis8 d8 e4 a,4 a8 gis8 
  fis4 gis8 a8 b4 b4 
  b4 dis4 gis,8 a8 b8 cis8 
  b4 b8 a8 gis4 e'4 
  e8 fis8 gis4 cis,8 d8 e4   %5
  fis8 gis8 a4 e4 cis4 
  b4 e,8 fis8 gis4 a4 
  b8 cis8 d4 cis4 
}

baixo = \relative c' {
  \partial 4 a4 
    a,8 b8 cis4 d8 e8 fis8 e8 
  dis4 e4 b4 b'8 a8 
  gis8 a8 gis8 fis8 e8 fis8 gis8 a8 
  b4 b,4 e4 e4 
  a8 gis8 fis8 eis8 fis8 e8 d8 cis8   %5
  b4 cis8 d8 e4 fis4 
  gis4 a4 e4 fis8 e8 
  dis4 e4 a,4 
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