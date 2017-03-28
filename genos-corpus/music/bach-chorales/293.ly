\version "2.10.33"

\header {
  title = "293 - Was Gott tut, das ist wohlgetan"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    g4 a4 b4 c4 
    d4. c8 b4 e4 
    d4 c4 b4 c4 
    a2 g4 } d'4 
  e4 e4 a,4 a4   %5
  d4 d4 g,4 b4 
  a4 g4 fis4 g4 
  e2 d4 d'4 
  c4 b4 a4 b4 
  a2 g4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    d4 d4 d4 g4 
    g4 fis4 d4 c8 b8 
    a4 e'4 d4 e4 
    d2 d4 } g4 
  g4 a8 g8 fis4 fis4   %5
  g4. f8 e4 d4 
  cis4 b4 a4 b4 
  a2 a4 g4 
  g8 a8 b8 c8 d4 d4 
  d2 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    b4 a4 g4 g8 a8 
    b4 a4 g4 g4 
    fis4 g8 a8 b8 fis8 g4 
    g4 fis4 d4 } b'4 
  c8 b8 a4 d4 d4   %5
  d8 c8 b8 d8 c4 b8 g8 
  e4 d4 d8 fis8 e8 d8 
  d4 cis4 fis4 d4 
  e4 d8 e8 fis4 g4 
  g4 fis4 g4   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'4 fis4 g4 e4 
    b4 d4 g,4 c4 
    d4 e8 fis8 g4 c,4 
    d2 g,4 } g'4 
  c,4 cis4 d4 c4   %5
  b4 b4 c4 g4 
  a4 b8 cis8 d4 g,4 
  a2 d,4 b'4 
  e8 fis8 g4 d8 c8 b8 g8 
  d2 g4   %10
  
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