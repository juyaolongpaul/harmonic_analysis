\version "2.10.33"

\header {
  title = "349 - Ich hab' in Gottes Herz und Sinn"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 fis8  g8 
    a4 b4 a4 d4 
    d4 cis4 d4 d4 
    cis4 b4 e4 d8 cis8 
    cis2 b4 } b4 
  b4 b8 cis16 d16 cis8 b8 a4   %5
  a4 gis4 a4 a4 
  b4 cis4 d4 cis4 
  b2 cis4 fis,8 g8 
  a4 b4 a4 d4 
  d4 cis4 d4 d4   %10
  cis4 b4 e4 d8 cis8 
  cis2 b4 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 d4 d8 e8 
    fis8 g8 e4 fis4 fis4 
    fis8 e8 d4 g4 fis8 gis8 
    ais8 b4 ais8 fis4 } fis4 
  e8 fis8 gis4 a4 e4   %5
  e4 e4 e4 fis4 
  fis4 fis4 fis4 e4 
  d4 cis8 b8 fis'4 d8 e8 
  fis4 g4 d8 e8 fis8 g8 
  a8 e8 fis8 g8 fis4 b4   %10
  e,8 fis8 g8 a8 b8 a16 g16 fis4 
  fis8 g8 fis8 e8 dis4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    a4 g4 fis8 g8 a8 b8 
    a4 a4 a4 b4 
    ais4 b4 b8 cis8 d8 e8 
    fis8 cis8 fis8 e8 d4 } d8 cis8 
  b4 e4 e8 d8 cis4   %5
  b4 cis8 d8 cis4 d8 cis8 
  b4 e4 d8 cis8 b8 ais8 
  fis'4 e4 ais,4 b4 
  cis4 d8 e8 fis4 a,4 
  a8 b8 a4 a4 a8 gis8   %10
  a4 b4 b8 cis8 d8 gis,8 
  ais8 b4 ais8 fis4 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    fis4 g4 d8 e8 fis8 g8 
    a4 a,4 d4 b4 
    fis'4 g8 fis8 e4 b'4 
    fis2 b,4 } b'8 a8 
  gis8 fis8 e4 a,8 b8 cis8 d8   %5
  e8 d8 e4 a,4 d8 e8 
  d8 cis8 b8 ais8 b4 cis4 
  d4 g4 fis4 b4 
  a4 g4 fis4 e8 d8 
  a'8 g8 a4 d,4 b4   %10
  a4 e'8 fis8 g8 a8 b4 
  fis8 e8 fis4 b,4 
  
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