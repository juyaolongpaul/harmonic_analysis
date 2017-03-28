\version "2.10.33"

\header {
  title = "104 - Wer nur den lieben Gott läßt walten"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    b4 cis4 d4 cis4 
    b4 cis4 ais2 
    fis4 a4 a4 g4 
    fis4 b4 b4 ais4 
    b2. } cis4   %5
  d4 e4 fis8 e8 fis8 g8 
  e4 e4 d4 fis4 
  e4 d4 cis4 b8 cis8 
  d8 e8 cis4 b2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  e8 
    fis4 fis4 fis8 g8 a4 
    g4 g4 fis2 
    cis4 cis8 d8 e4 e4 
    fis8 e8 d4 cis8 d8 e4 
    d2. } fis4   %5
  fis8 b8 a8 g8 fis4 b4 
  b4 a8 g8 fis4 a4 
  ais4 b4 e,8 dis8 e4 
  fis8 g8 fis8 e8 dis2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  cis8 
    d4 cis4 b4 cis8 d8 
    e4 e8 d8 cis2 
    ais4 a4 a8 b8 cis4 
    cis4 b8 a8 g4 fis4 
    fis2. } ais4   %5
  b8 g'8 fis8 e8 d4 d4 
  d8 b8 cis4 a4 a8 b8 
  cis4 fis,8 g8 a8 fis8 b4 
  b4 ais4 fis2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 b4 
    b'4 ais4 b4 fis4 
    g8 fis8 e4 fis2 
    fis,4 fis'4 cis4 b8 a8 
    d4 g8 fis8 e4 fis4 
    b,2. } fis'4   %5
  b4 cis4 d8 cis8 b8 a8 
  g4 a4 d,4 d4 
  cis4 b4 a4 g4 
  fis'8 e8 fis4 b,2 
  
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