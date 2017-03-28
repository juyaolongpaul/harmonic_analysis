\version "2.10.33"

\header {
  title = "112 - Wer nur den lieben Gott läßt walten"
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
    fis2 r4 a4 
    a4 g4 fis4 b4 
    b4 ais4 b4 } cis4   %5
  d4 e4 fis4 fis4 
  e4. d8 d4 fis4 
  e4 d4 cis4 b8 cis8 
  d4 cis4 b2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  e8 
    fis4 fis4 fis4 fis8 e8 
    d4 g4 fis2 
    cis2 r4 fis4 
    fis4 e4 d4 d4 
    g4 fis4 fis4 } ais4   %5
  b4 a4 a4 b4 
  b4 a4 a4 a4 
  ais4 b4 b8 ais8 b4 
  b4 ais4 fis2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  cis8 
    d4 cis4 b4 ais4 
    b4 e8 d8 cis2 
    ais2 r4 cis4 
    d4 a4 a4 b4 
    cis4 cis4 d4 } fis4   %5
  fis4 e4 d4 d4 
  d4 cis4 fis4 d4 
  e4 fis4 g8 fis8 fis8 e8 
  fis4. e8 dis2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 b4 
    b'4 ais4 b4 fis4 
    g4 e4 fis2 
    fis,2 r4 fis'4 
    b,4 cis4 d4 g8 fis8 
    e4 fis4 b,4 } fis'4   %5
  b4 cis4 d8 cis8 b8 a8 
  g4 a4 d,4 d'4 
  cis4 b4 e,8 fis8 g4 
  fis8 e8 fis4 b,2 
  
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