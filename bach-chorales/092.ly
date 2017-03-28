\version "2.10.33"

\header {
  title = "92 - O Jesu Christ, du höchstes Gut"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    b4 ais4 b4 cis4 
    d4 cis4 b4 b4 
    cis4 cis4 fis4 e4 
    d2 cis4 } fis4 
  e4 d4 cis4 e4   %5
  d4 cis4 b4 d4 
  cis4 b4 a8 gis8 fis8 gis8 
  a4 b4 cis4 cis4 
  d4 e4 fis4. e8 
  d4 e8 d8 cis2   %10
  b2.
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    e8 fis8 g4 fis4 g4 
    fis4 fis8 e8 d4 e4 
    e8 fis8 g4 fis4 cis'4 
    cis8 ais8 b4 ais4 } a4 
  a8 gis8 a8 b8 a4 g8 fis8   %5
  fis4 fis4 fis4 fis8 gis8 
  a4 gis4 fis4 cis4 
  fis4 gis4 a4 a8 g8 
  fis8 b8 a8 g8 fis2~ 
  fis4 e8 fis8 g4 fis8 e8   %10
  dis2.
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    cis8 d8 e4 d8 cis8 b8 ais8 
    b8 cis8 ais4 b4 gis4 
    a4 e'4 d8 e8 fis4 
    fis4. eis8 fis4 } fis4 
  b,4 fis'8 e8 e8 d8 cis4   %5
  cis8 b8 b8 ais8 d4 b4 
  e4 d8 cis8 cis8 b8 a8 gis8 
  fis8 a8 d4 e4 e4 
  d8 g8 fis8 e8 d4 cis4~ 
  cis8 fis,8 b2 ais4   %10
  fis2. 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    cis4 cis,4 d4 e4 
    fis8 e8 fis4 b,4 e4 
    a,4 a'4 d8 cis8 b8 ais8 
    b4 b,4 fis'4 } d4 
  e4 fis8 gis8 a4 ais4   %5
  b4 fis4 b,4 b'4 
  a8 gis8 fis8 eis8 fis4. e8 
  d8 cis8 b4 a4 a'4 
  b4 cis4 d8 cis8 b8 ais8 
  b8 a8 g8 fis8 e4 fis4   %10
  b,2.
  
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