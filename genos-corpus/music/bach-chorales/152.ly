\version "2.10.33"

\header {
  title = "152 - Meinen Jesum laß ich nicht, weil er sich für mich gegeben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  a4 a4 b4 b4 
  cis4 cis4 d2 
  e4 e4 d4 d4 
  cis4 b8 cis16 d16 b2 
  a2 a4 a4   %5
  b4 b4 a4 g4 
  fis2 a4 a4 
  g4 g4 fis4 e8 fis16 g16 
  e2 d2 
  d'4 d4 cis4 cis4   %10
  b4 b4 a2 
  a4 a4 g4 fis4 
  e4 e4 d2 
  
}

alto = \relative c' {
  fis8 g8 a4 a4 g4 
  g8 e8 a8 g8 fis2 
  a4 a4 a4 gis8 fis8 
  e4 e4 fis4 e4 
  e2 fis8 g8 a4   %5
  a4 g4 g8 fis8 fis8 e8 
  d2 cis4 fis4 
  fis8 e16 d16 e4. d8 d4 
  d4 cis4 a2 
  fis'4 fis4 fis4 e4   %10
  fis4 e4 e2 
  fis4 fis4 fis8 e4 d8 
  d8 cis16 b16 cis4 a2 
  
}

tenor = \relative c' {
  d4 d4 d4 e4 
  e4 a,4 a2 
  cis4 cis4 cis4 b4 
  b8 a8 a4 a4 gis4 
  cis2 d4 d4   %5
  d4 e4 e8 cis8 d8 a8 
  a2 a4 b4 
  b4 a4 a4 b4 
  a4 g4 fis2 
  b4 b4 a4 a4   %10
  a4 gis4 cis2 
  cis4 d4 d8 a8 a8 b8 
  b8 g8 e8 a8 fis2 
  
}

baixo = \relative c {
  d8 e8 fis8 d8 g8 fis8 g8 e8 
  a8 g8 fis8 e8 d2 
  a8 b8 cis8 a8 b8 cis8 d8 b8 
  cis8 d8 e8 cis8 d8 b8 e4 
  a,2 d8 e8 fis8 d8   %5
  g8 fis8 e8 d8 cis8 a8 b8 cis8 
  d2 fis8 e8 dis8 b8 
  e8 d8 cis8 a8 d8 fis8 g8 e8 
  a8 g8 a8 a,8 d2 
  b8 cis8 d8 e8 fis8 gis8 a8 fis8   %10
  dis8 b8 e4 a,2 
  fis'8 e8 d8 cis8 b8 cis8 d8 b8 
  g8 e8 a4 d,2 
  
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