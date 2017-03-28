\version "2.10.33"

\header {
  title = "28 - Nun komm, der Heiden Heiland"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  b4 b4 a4 d4 
  cis8 b8 cis4 b2 
  b8 cis8 d4 e4 d4 
  e4 fis4 d2 
  d4 e4 fis8 e8 d4   %5
  e8 d8 cis4 b2 
  b4 b4 a4 d4 
  cis8 b8 cis4 b2 
  
}

alto = \relative c' {
  fis4 e4 fis4 fis8 g16 a16 
  g8 fis8 e4 d2 
  fis4 fis8 gis8 a4 a8 b8 
  cis4 d8 fis,8 g2 
  a4 a4 a8 ais8 b4   %5
  cis8 b8 ais4 fis2 
  fis4 eis4 fis8 gis8 a8 b8 
  fis8 b4 ais8 fis2 
  
}

tenor = \relative c' {
  d4 cis8 b8 cis4 b4 
  b4. ais8 fis2 
  d'8 cis8 b4 cis4 d4 
  g,4 a8 b16 c16 b2 
  a4 e'4 d8 cis8 b4   %5
  g'4 cis,8 fis8 d2 
  d4 cis8 b8 cis4 fis4 
  e8 d8 cis16 d16 e8 dis2 
  
}

baixo = \relative c' {
  b8 a8 g4 fis8 e8 d8 b8 
  e8 d8 e8 fis8 b,2 
  b4 b'4 a8 g8 fis4 
  e4 d4 g2 
  fis8 e8 d8 cis8 d4 g8 fis8   %5
  e4 fis4 b,2 
  b'8 a8 gis4 fis8 e8 fis8 gis8 
  ais8 b8 fis4 b,2 
  
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