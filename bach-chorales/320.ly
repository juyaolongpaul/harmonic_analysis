\version "2.10.33"

\header {
  title = "320 - Gott sei uns gnädig und barmherzig"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key fis \minor
}


soprano = \relative c'' {
  cis2 e2 
  cis4 cis4 cis4 cis4 
  d2 cis2 
  b1 
  a1   %5
  cis2 e2 
  b4 b4 b4 b4 
  b2 fis4 gis4 
  a2 gis2 
  fis1~   %10
  fis1 
  
}

alto = \relative c' {
  fis2 e2 
  e4 eis4 eis gis
  fis4 e2 fis4 
  fis2 e2 
  e1   %5
  e1 
  e4 gis4 fis4 a4 
  gis2 fis2 
  fis2. eis4 
  cis2 d2~   %10
  d2 cis2 
  
}

tenor = \relative c' {
  a2 b2 
  cis4 b4 a4 gis4 
  a4 b2 a4 
  a2 gis2 
  a1   %5
  a2 a2 
  gis4 b4 a4 fis'4 
  e2 b2 
  cis2 b2 
  ais2 b2~   %10
  b2 ais2 
  
}

baixo = \relative c {
  fis2 gis2 
  a4 gis4 fis4 eis4 
  fis4 gis4 a4 fis4 
  d4 b4 e2 
  a,1   %5
  a4 b4 cis4 d4 
  e2 dis2 
  e2 d2 
  cis1 
  fis,4 e'4 d4 b4   %10
  fis1 
  
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