\version "2.10.33"

\header {
  title = "58 - Herzlich lieb hab' ich dich, o Herr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    cis4 b4 a4 g8 fis8 
    b4 b4 a4 d4 
    cis4 b4 a4 g8 fis8 
    b4 b4 a4 a4 
    b4 cis4 d8 e8 fis4     %5
    e8 d8 e4 d4 } d4 
  d4 e4 d4 b8 cis8 
  d4 d4 cis4 cis4 
  d4 e4 d4 b8 cis8 
  d4 d4 cis4 fis4   %10
  g8 fis8 e4 fis8 e8 d4 
  d4 cis4 d4 cis4 
  d4 cis4 b4 a4 
  a4 gis4 a4 fis'4 
  e4 d4 cis4 fis4   %15
  g8 fis8 e4 fis8 e8 d4 
  d4 cis4 d4 d4 
  d4 d4 e4 d4 
  d4 cis4 d
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    e8 fis8 g8 fis8 e4 d4 
    d4 d4 d4 fis8 eis8 
    fis8 e8 d8 e8 fis8 e8 d8 fis8 
    fis4 f4 fis4 fis8 e8 
    d8 b'8 a8 g8 fis8 g8 a8 b8     %5
    e,8 fis8 g4 fis4 } a4 
  a8 g4 fis8 fis4 e4 
  e4 e4 e4 e4 
  fis4 g4 fis4 e4 
  d8 e8 fis4 fis4 fis4   %10
  e4 e4 a8 g8 fis4 
  e4 e4 fis4 g4 
  fis4 e8 fis8 gis4 fis4 
  fis4 e4 e4 d'4 
  ais4 b4 ais4 cis4   %15
  b4 b8 a8 a4 a4 
  e4 fis4 fis4 a4 
  a4 g8 fis8 g4 fis4 
  e8 d8 e4 fis
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  b8 
    cis4 d4 e8 a,8 a4 
    a4 g4 fis4 fis8 gis8 
    a4 b4 cis4 d8 cis8 
    b4 cis4 cis4 d8 cis8 
    b8 g'8 fis8 e8 d4 d4     %5
    d4 cis4 a4 } fis'8 e8 
  d4 cis4 b8 cis8 d4 
  gis,8 a8 b4 a4 a4 
  a4 a4 a4 g4 
  fis4 b4 ais4 b4   %10
  b4 cis4 cis4 d8 cis8 
  b4 a4 a4 e'4 
  a,8 b8 cis4 d4 cis4 
  b4 b4 cis4 a8 b8 
  cis4 fis,4 fis'4 fis8 e8   %15
  d4 e4 d4 d8 cis8 
  b4 cis4 b4 d8 c8 
  b4 b4 b8 a8 a4 
  a4 a4 a
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d4 
    a4 b4 cis4 d4 
    g,8 a8 b8 cis8 d4 b4 
    fis'4 g4 a4 b8 a8 
    gis4 cis,4 fis4 d4 
    g4 a4 b4 a8 g8     %5
    a4 a,4 d4 } d'8 cis8 
  b4 ais4 b8 a8 gis8 fis8 
  e8 fis8 gis8 e8 a4 a8 g8 
  fis8 e8 d8 cis8 d4 e4 
  b8 cis8 d8 e8 fis4 dis4   %10
  e4 a8 g8 fis4 b8 a8 
  gis4 a4 d,4 e4 
  fis8 gis8 a4 eis4 fis8 eis8 
  dis4 e4 a,4 d4 
  cis4 b4 fis'4 ais4   %15
  b4 cis4 d4 fis,4 
  gis4 ais4 b4 fis4 
  g8 a8 b4 cis,4 d4 
  a'4 a,4 d
  
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