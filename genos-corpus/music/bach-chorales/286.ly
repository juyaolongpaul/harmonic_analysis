\version "2.10.33"

\header {
  title = "286 - Befiehl du deine Wege"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    b4 a4 g4 fis4 
    e2 fis4 cis'4 
    d4 d4 cis8 b8 cis4 
    b2. } d4 
  cis8 b8 a4 b4 cis4   %5
  d2 d4 a4 
  b4 a4 g8 fis8 g4 
  fis2. d'4 
  cis8 d8 e4 d4 cis4 
  b2 cis4 fis,4   %10
  g4 fis4 e4 a4 
  fis2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d8 e8 fis4 b,8 cis8 d4 
    d4 cis8 b8 cis4 fis4 
    fis4 fis4 fis4. e8 
    d2. } fis4 
  e4 e8 d8 cis4 g'4   %5
  a4 g4 fis4 fis4 
  fis8 e8 e8 dis8 e4 e4 
  dis2. e4 
  e4 e4 fis8 e4 gis8 
  fis4 e4 e4 d4   %10
  d4 d4 e4 e4 
  e4 d4 cis4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    b8 cis8 d4 d8 cis8 cis8 b8 
    b8 ais8 b4 ais4 ais4 
    b4 b4 b4 ais4 
    fis2. } b8 a8 
  g4 a4 g8 b8 e4   %5
  d8 c8 b4 a4 d8 c8 
  b4 c8 b8 b4 b4 
  b2. b4 
  a4. gis8 a8 b8 cis4 
  d8 b8 gis4 a4 a4   %10
  b4 a4 a4 cis4 
  cis8 ais8 b8 gis8 ais4 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  a8 
    g4 fis4 e4 d4 
    g2 fis4 fis4 
    b,8 cis8 d8 e8 fis4 fis4 
    b,2. } b'4 
  e,4 fis4 g4 fis8 e8   %5
  fis4 g4 d4 d4 
  g4 fis4 e8 g8 fis8 e8 
  b'2. gis4 
  a8 b8 cis4 fis,8 gis8 a4 
  d,4 e4 a,4 d8 cis8   %10
  b8 cis8 d4. cis8 b8 cis8 
  d8 cis8 b4 fis'4 
  
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