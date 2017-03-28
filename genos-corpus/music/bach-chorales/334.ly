\version "2.10.33"

\header {
  title = "334 - Für deinen Thron tret' ich hiermit"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \major
}


soprano = \relative c'' {
  \partial 4 d4 
  d2 cis4 
  b2 a4 
  d4 e2 
  fis2 fis4 
  fis2 fis4   %5
  e2 fis4 
  g4 fis2 
  e2 d4 
  e2 fis4 
  e2 d4   %10
  b4 cis2 
  d2 a'4 
  fis2 d4 
  e2 g4 
  fis4 e2   %15
  d2
  
}

alto = \relative c'' {
  \partial 4 a4 
  b2 a4 
  g2 fis4 
  fis4 b4 a4 
  a2 cis4 
  b2 b4   %5
  b4 a4 a4 
  a4 d4 a4 
  a2 b4 
  a2 a4 
  b4 a8 g8 fis4   %10
  b4 b4 ais4 
  b2 a4 
  a2 b4 
  a2 a4 
  a4 g2   %15
  fis2
  
}

tenor = \relative c' {
  \partial 4 fis4 
  fis2 fis4 
  d2 d4 
  d2 cis4 
  d2 cis4 
  d2 d4   %5
  d4 cis4 d4 
  e4 a,4 d4 
  cis2 fis4 
  e2 d4 
  d4 cis4 d4   %10
  d4 g4 fis4 
  fis2 e4 
  fis2 fis4 
  e2 cis4 
  a4 b4 cis4   %15
  a2
  
}

baixo = \relative c {
  \partial 4 d4 
  b4 d4 fis4 
  g4 b4 d4 
  b4 g4 a4 
  d,2 ais'4 
  b2 a4   %5
  gis4 a4 fis4 
  cis4 d4 fis4 
  a2 b4 
  cis2 d4 
  g,4 a4 b4   %10
  g4 e4 fis4 
  b,2 cis'4 
  d4 cis4 b4 
  cis4 b4 a4 
  d4 g,4 a4   %15
  d,2
  
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