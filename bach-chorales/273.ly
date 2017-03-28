\version "2.10.33"

\header {
  title = "273 - Ein' feste Burg ist unser Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 a8 b8 cis4 
    d8 cis8 b4 a4 d4 
    cis4 b4 a4 b8 a8 
    g8 fis8 e4 d4 } d4 
  a'4 b4 a4 gis4   %5
  a2 d,4 a'4 
  b4 cis4 d4 cis4 
  d4 cis4 b2 
  a4 b4 b4 a4 
  b8 a8 g4 fis4 d'4   %10
  cis4 b4 a4 b8 a8 
  g8 fis8 e4 d4 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a8 g8 fis8 e8 d4 e4 
    a,8 a'4 gis8 e4 b'4 
    e,8 fis8 g4 fis8 e8 d8 cis8 
    b4 cis4 a4 } d4 
  cis8 fis8 e4 e8 fis8 e4   %5
  e2 d4 d8 c8 
  b8 d8 g4 fis4 fis4 
  fis4 e8 a8 fis4 e8 d8 
  cis4 e4 d8 e8 fis4 
  fis4 fis8 e8 dis4 fis4   %10
  e4 d8 g8 g8 fis16 e16 fis4 
  g8 b,8 cis4 a4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 fis8  e8 
    d8 e8 fis8 g8 a4 g4 
    fis4 b,8 e8 cis4 fis,8 gis8 
    a4 e'8 d8 cis4 b8 cis8 
    d4 a8 g8 fis4 } fis8 g8 
  a8 d8 cis8 b8 cis4 b4   %5
  cis2 fis,8 g8 a4 
  d4 e4 a,4 a4 
  a8 gis8 a4 a4 gis4 
  e4 g8 a8 b8 cis8 d8 c8 
  b4 b4 b4 b8 a8   %10
  g8 a8 b4 e8 cis8 d4 
  d4 a8 g8 fis4 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  cis8 
    b4 a8 g8 fis8 g8 fis8 e8 
    fis4 e4 a,4 b4 
    cis8 d8 e4 fis4 g8 a8 
    b8 g8 a8 a,8 d4 } d8 e8 
  fis4 gis4 a8 d,8 e4   %5
  a,2 b4 fis'4 
  g4 e4 d4 fis4 
  b,4 cis4 d4 e4 
  a,4 e'8 fis8 g4 d4 
  dis4 e4 b4 b4   %10
  e8 fis8 g8 e8 cis8 a8 d8 cis8 
  b8 g8 a4 d,4 
  
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