\version "2.10.33"

\header {
  title = "17 - Erschienen ist der herrliche Tag"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key e \minor
}


soprano = \relative c' {
  e4 e4 e4 
  b'2 cis4 
  d4 cis4 b4 
  a2.\fermata 
  b4 cis4 dis4   %5
  e2 b4 
  d4 cis2 
  b2\fermata b4 
  d4 c4 b4 
  a4 g4 fis4   %10
  g4 e2 
  d2\fermata d4 
  g2 a4 
  b2 a4 
  g4 d'2   %15
  b2\fermata a4 
  g4 fis2 
  e2.\fermata 
  
}

alto = \relative c' {
  b4 b4 cis4 
  d4 e4 e4 
  fis8 gis8 a4 gis4 
  e2. 
  e4 e4 fis4   %5
  g8 a8 g8 fis8 e4 
  fis4 g4 fis4 
  dis2 e4 
  d2 d4 
  e2 d4   %10
  d2 cis4 
  a2 b4 
  b4 e4 d4 
  d8 e8 fis2 
  e4 fis8 g8 a4   %15
  g2 fis4 
  e2 dis4 
  b2. 
  
}

tenor = \relative c' {
  g8 a8 b4 a4 
  fis4 b4 a4 
  a4 fis4 b4 
  cis2. 
  b4 a4 a4   %5
  b2 b4 
  b2 ais4 
  fis2 g4 
  a2 g4 
  e4 a2   %10
  g8 a8 b4 a8 g8 
  fis2 fis4 
  g4 c2 
  b4 b2 
  b8 c8 d2   %15
  d2 c4 
  b4 a8 g8 a4 
  gis2. 
  
}

baixo = \relative c {
  e8 fis8 g4 a4~ 
  a4 gis4 a4 
  fis4 d4 e4 
  a,2.\fermata
  gis'4 a4 fis4   %5
  e8 dis8 e8 fis8 g4 
  fis4 e4 fis4 
  b,2\fermata e4 
  fis2 g4 
  cis,2 d4   %10
  b4 g4 a4 
  d2\fermata b4 
  e2 fis4 
  g4 dis2 
  e4 b'4 fis4   %15
  g2\fermata a4 
  b4 b,2 
  e2.\fermata
  
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