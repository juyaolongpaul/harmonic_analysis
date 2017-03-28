\version "2.10.33"

\header {
  title = "65 - Was Gott tut, das ist wohlgetan"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    g4 a4 b4 e4 
    d4. c8 b4 e4 
    d4 c4 b4 c8 d8 
    a2 g4 } d'4 
  e4 e4 a,4 a4   %5
  d4 d4 g,4 b4 
  a4 g4 fis4 g8 fis8 
  e2 d4 d'4 
  c4 b4 a4 b8 c8 
  a2 g2   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 b8  c8 
    d4 d4 d4 e8 fis8 
    g4 fis4 g4 g4 
    fis4 g8 a8 b8 a8 g4 
    g4 fis4 d4 } g4 
  g4 g4 fis4 fis8 g8   %5
  a8 g8 fis4 e4 e4 
  e4 d8 cis8 d4 d4 
  d4 cis4 d4 g8 fis8 
  e4 d8 e8 fis4 g4 
  g4 fis4 d2   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g8  a8 
    b4 a4 g4 g8 a8 
    b4 a8 d8 d4 c8 b8 
    a4 e'4 d4 e4 
    e4 d8 c8 b4 } b4 
  c8 b8 a4 d4 d4   %5
  a4 b4 b4 g4 
  a4 a4 a4 g8 a8 
  b4 a8 g8 fis4 g4 
  g8 a8 b8 c8 d4 d4 
  d8 c16 b16 c8 a8 b2   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'4 fis4 g4 c,4 
    b8 c8 d4 g,4 c4 
    d4 e8 fis8 g8 fis8 e8 d8 
    c4 d4 g,4 } g'4 
  c,4 cis4 d4 d8 e8   %5
  fis4 b,4 e4 e8 d8 
  cis8 b8 a4 d8 cis8 b8 a8 
  g4 a4 d4 b4 
  e8 fis8 g4 d8 c8 b8 g8 
  d'2 g,2   %10
  
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