\version "2.10.33"

\header {
  title = "336 - Wo Gott der Herr nicht bei uns hält"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 b8 cis8 d8 e8 fis4 
    e4 e4 d4 d4 
    cis4 b8 cis8 d4 e8 d8 
    cis2 b4 } e4 
  fis4 fis4 cis8 d8 e4   %5
  d4 d4 cis4 d4 
  e4 b8 cis8 d4 cis4 
  b4 b4 a4 d4 
  cis4 b8 cis8 d4 e8 d8 
  cis2 b4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 fis8  g8 
    a4 g4 a4 a4 
    b4 a8 g8 fis4 gis4 
    a4 g4 fis4 e8 fis8 
    g4 fis4 fis4 } a4 
  a4 a4 e8 fis8 g4   %5
  fis4 gis4 ais4 b4 
  b8 a8 gis4 fis4 e4 
  a4 gis4 e4 gis4 
  a4 gis8 ais8 b4 b8 fis8 
  g4 fis8 e8 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 d4 d4 
    d4 cis4 a4 b4 
    e4 e4 d8 cis8 b4 
    b4 ais8 e'8 d4 } ais4 
  a4 a8 b8 cis4 cis4   %5
  d8 cis8 b4 fis'4 fis4 
  b,4 e4 a,8 b8 cis4 
  fis4 e8 d8 cis4 d4 
  e4 e4 fis4 b,4 
  b4 ais4 fis4   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    fis4 g4 fis8 e8 d4 
    gis4 a4 d,4 b4 
    a4 e'4 b'8 a8 g8 fis8 
    e4 fis4 b,4 } cis4 
  d8 e8 fis8 g8 a4 ais4   %5
  b4 eis,4 fis4 b8 a8 
  gis8 fis8 e4 fis8 gis8 a4 
  dis,4 e4 a,4 b4 
  cis8 d8 e4 b'8 a8 g8 fis8 
  e4 fis4 b,4   %10
  
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