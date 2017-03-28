\version "2.10.33"

\header {
  title = "20 - Ein' feste Burg ist unser Gott"
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
    cis4 b4 a4 b4 
    g8 fis8 e4 d4 } d4 
  a'4 b4 a4 gis4   %5
  a2 d,4 a'4 
  b4 cis4 d4 cis4 
  d8 cis8 b4 a4 b4 
  b4 a4 b8 a8 g4 
  fis4 d'4 cis4 b4   %10
  a4 b8 a8 g4 fis4 
  e2 d4 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    d,4 d8 e8 fis4 g4 
    a4 gis4 e4 b'4 
    e,4 d4 cis4 b4 
    cis8 d4 cis8 d4 } d4 
  cis4 d4 e8 d8 b4   %5
  cis2 d8 e8 fis4 
  g8 fis8 e4 fis4 fis4 
  fis16 gis16 a4 gis8 e4 g4 
  g4 dis8 e8 fis4 e4 
  dis4 fis4 fis8 e8 d8 e8   %10
  fis8 e8 d4 e8 a,8 d4~ 
  d4 cis4 a4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    fis4 b,8 cis8 d4 e4 
    fis4. e16 d16 cis4 fis,8 gis8 
    ais4 b4 fis'8 e8 d4 
    e8 a,4 g8 fis4 } fis8 g8 
  a4 gis8 fis8 e8 fis8 e4   %5
  e2 a4 d4 
  d4 a4 a4 a4 
  b8 e8 fis8 e8 cis4 e4 
  d8 e8 fis4 b,4 b4 
  b4 b4 a4 b8 cis8   %10
  d4 g,8 fis8 e4 fis16 g16 a8 
  b4 a8 g8 fis4 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  cis8 
    b4 a8 g8 fis4 e4 
    d4 e4 a,4 b4 
    cis4 d8 e8 fis4 g8 fis8 
    e8 d8 a4 d4 } d8 e8 
  fis4 e8 d8 cis8 d8 e4   %5
  a,2 fis'4 e8 d8 
  g4 a4 d,4 fis4 
  b,8 cis8 d8 e8 a,4 e'8 fis8 
  g4 fis8 e8 dis4 e4 
  b4 b4 fis'4 g4   %10
  d8 cis8 b4 cis4 d4 
  g,4 a4 d,4 
  
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