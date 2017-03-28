\version "2.10.33"

\header {
  title = "250 - Ein' feste Burg ist unser Gott"
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
  a4 d,4 a'4 b8 cis8 
  d4 cis4 d8 cis8 b b 
  a4 b4 b4 a4 
  b8 a8 g4 fis4 d'4 
  cis4 b4 a4 b8 a8   %10
  g8 fis8 e4 d 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a8 g8 fis8 g8 a8 fis8 g8 a8 
    fis8 e8 d8 e8 cis4 fis4 
    e8 fis8 g4 fis8 e8 d4 
    d4 cis4 a4 } d4( 
  cis8) fis8 e4 cis8 d8 b4   %5
  a4 d4 d4 g8 g8 
  fis4 fis4 fis16 gis16 a8 a8 gis8 
  a4 g4 g4 fis4 
  fis4 e4 dis4 fis8 gis8 
  a4 g4 g4 fis8 e8   %10
  d4 cis a
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 fis8  e8 
    d4 c8 b8 d4 e4 
    a,4 a8 gis8 a4 fis8 gis8 
    a4 e'8 d8 cis4 b8 cis8 
    d4 a8 g8 fis4 } fis8 gis8 
  a8 d8 cis8 b8 e8 fis8 e8 d8   %5
  cis4 fis,8 g8 a4 e'8 e8 
  d4 a4 b8 e8 fis8 e8 
  cis4 b8 cis8 d4 d8 c8 
  b4 b4 b4 b4 
  e4 e4 e4 d8 a8   %10
  b4 e,8 a8 fis4
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  cis8 
    b4 a8 g8 fis4 e4 
    d4 e4 a,4 b4 
    cis8 d8 e4 fis4 g8 a8 
    b8 g8 a8 a,8 d4 } d8 e8 
  fis4 gis4 a8 d,8 e4   %5
  a,4 b4 fis'4 e8 a,8 
  d4 fis4 b,8 cis8 d8 e8 
  a,4 e'4 b8 cis8 d4 
  dis4 e b b
  cis8 d8 e8 d8 cis8 a8 d8 cis8   %10
  b8 g8 a4 d
  
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