\version "2.10.33"

\header {
  title = "78 - Herzliebster Jesu, was hast du"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  \partial 4 b4 
  b4 b4 ais8 gis8 fis4 
  b4 cis4 d4 d4 
  e4 d4 cis4 cis4 
  d4 e4 fis8 e8 d4 
  g4 g4 fis8 e8 fis4   %5
  e2 d4 d4 
  cis4 b4 a4 fis8 g8 
  a4 a4 b4 a4 
  g2 fis4 fis'4 
  e4 d4 cis2   %10
  b2.
  
}

alto = \relative c' {
  \partial 4 fis4 
  g4 d8 e8 fis4 cis4 
  fis4 fis4 fis4 fis4 
  fis2 fis4 fis4 
  fis4 a4 a4 b8 a8 
  g4 a4 a4 d8 cis8   %5
  b4 a8 g8 fis4 fis4 
  e4 d4 e4 d8 e8 
  fis4 fis4 g4 fis4 
  fis8 dis8 e4 e4 d4 
  g4 fis4 fis4. e8   %10
  dis2. 
  
}

tenor = \relative c' {
  \partial 4 d4 
  e4 b4 cis8 b8 ais4 
  b4 ais4 b8 cis8 d4 
  cis4 b4 ais4 ais4 
  b4 cis4 d8 cis8 b8 cis8 
  d4 e4 e4 d4   %5
  d4 cis4 a4 b8 a8 
  g4 g8 fis8 e8 a8 a4 
  d4 d4 d4 c4 
  b4. ais16 b16 cis4 b4 
  b8 ais8 b4 b4 ais4   %10
  fis2.
  
}

baixo = \relative c' {
  \partial 4 b4 
  e,8 fis8 g4 fis4 fis8 e8 
  d4 fis4 b,4 b'4 
  ais4 b4 fis4 fis4 
  b4 a4 d,4 g8 a8 
  b4 cis4 d8 cis8 b8 a8   %5
  g4 a4 d,4 b4 
  e8 fis8 g4 cis,4 d4 
  d'8 c8 b8 a8 g8 fis8 e8 dis8 
  e2 ais,4 b4 
  cis4 d8 e8 fis2   %10
  b,2.
  
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