\version "2.10.33"

\header {
  title = "285 - Wär' Gott nicht mit uns diese Zeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    c8 b8 a8 b8 c8 d8 e4 
    d4 d4 c4 c4 
    b4 a8 b8 c4 d4 
    b2 a4 } d4 
  e4 e4 b8 c8 d4   %5
  c4 c4 b4 c4 
  d4 a8 b8 c4 b4 
  a4 a4 g4 c4 
  b4 a8 b8 c4 d8 c8 
  b2 a4   %10
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    g4 f4 e8 d8 c4 
    d8 e8 f4 e4 fis4 
    g4 fis8 gis8 a4 a8 g8 
    f4 e8 d8 c4 } g'4 
  g4 g4 d4. e8   %5
  fis8 gis8 a4 gis4 a4 
  a8 g8 fis4 e4 d8 e8 
  f4 f4 e4 a4 
  g4 a4 gis8 b8 a8 g8 
  f4 e8 d8 cis4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e8 d8 c4 c8 b8 a8 g8 
    a4 b4 c4 c4 
    d4 d4 e4 a,4 
    a4 gis4 a4 } g4 
  c4 g8 a8 b4 a8 b8   %5
  c8 d8 e4 e4 e4 
  d4 d4 g,8 a8 b4 
  c8 b8 c8 d8 e4 ees4 
  d4 d4 e4 a,4 
  a4 gis4 e4   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    e4 f8 g8 a8 b8 c4 
    f,4 g4 c,4 a'4 
    g4 d'4 a8 g8 f8 e8 
    d4 e4 a,4 } b4 
  c8 d8 e8 f8 g4 fis8 gis8   %5
  a8 b8 c8 d8 e4 a,8 g8 
  fis8 e8 d4 e8 fis8 g4 
  a8 g8 a8 b8 c4 fis,4 
  g4 f4 e8 c8 f8 e8 
  d8 b8 e4 a,4   %10
  
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