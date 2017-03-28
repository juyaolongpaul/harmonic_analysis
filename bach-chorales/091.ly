\version "2.10.33"

\header {
  title = "91 - Verleih' uns Frieden gnädiglich"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key fis \minor
}


soprano = \relative c' {
  \partial 4 fis4 
  fis4 fis4 eis4 fis8 gis8 
  a4 gis4 fis4 a4 
  b4 b4 cis8 b8 a4 
  b2 cis4 cis4 
  cis4 b4 cis8 b8 a4   %5
  b8 a8 gis4 fis4 fis8 gis8 
  a4 a4 b4 a8 gis8 
  fis2 fis4 b4 
  cis4 b4 fis8 gis8 a4 
  gis8 fis8 gis4 fis4 fis4   %10
  fis4 e4 a4 b4 
  cis8 b8 a4 b4 b4 
  cis2 d2 
  cis4 b8 cis16 d16 b4. a8 
  a2. cis4   %15
  cis8 b8 a4 b8 cis8 d4 
  cis2 e4 cis4 
  d4 cis8 b8 cis4 cis4 
  d4 cis8 d8 e8 d8 cis4 
  b2 a2   %20
  cis4 cis4 a4 b4 
  a4 gis4 fis2 
  e4 fis8 gis8 a4 fis4 
  fis2. e4 
  fis4 gis4 a4 gis4   %25
  fis2. eis4 
  fis2. 
  
}

alto = \relative c' {
  \partial 4 cis4 
  d4 d4 cis2 
  cis8 fis8 fis8 eis8 cis4 fis4 
  e8 fis8 gis4 a2 
  a8 gis16 fis16 gis4 a4 a4 
  a4 eis8 fis8 gis4 fis4   %5
  fis4 e4 dis4 dis8 eis8 
  fis4 fis4 gis8 fis8 eis4 
  fis8 e8 d4 cis4 e4 
  e4 e4 d8 e8 fis4 
  fis4 eis4 cis4 cis4   %10
  cis4 cis4 d4 d4 
  cis4 cis4 fis4 e4 
  e2 b'2 
  e,4 fis4 e4 e4 
  e2. gis4   %15
  a8 gis8 fis4 fis4 fis8 eis8 
  fis2 gis8 b8 b8 a8 
  a8 gis8 a8 gis8 a8 b8 cis4 
  cis8 b8 a4 e4 e4 
  fis4 e8 d8 cis2   %20
  e4 e4 fis4 fis4 
  fis4 e4 e4 d4 
  cis4 d4 e4 d4 
  d2. cis4 
  cis2. d8 e8   %25
  fis4 gis8 a8 d,4 cis4 
  cis2. 
  
}

tenor = \relative c' {
  \partial 4 a4 
  a4 a4 gis8 b8 a8 gis8 
  fis4 cis'8 b8 a4 cis4 
  b4 e4 e4. d16 cis16 
  b4 e4 e4 e4 
  e8 d8 d4 cis4 cis4   %5
  b4 b4 b4 b4 
  cis4 d4 d4 cis8 b8 
  a8 ais8 b4 a4 gis4 
  a4 b8 cis8 d4 cis4 
  cis4. b8 a4 a4   %10
  a4 gis4 fis4 gis4 
  a8 gis8 a4 a4 gis4 
  a2 gis2 
  a4 a4 a8 fis8 gis8 d'8 
  cis2. cis4   %15
  fis4. e8 d8 cis8 b8 gis8 
  a2 b4 e4 
  e4 e8 d8 cis8 b8 a8 gis8 
  fis8 gis8 a4 b4 a4 
  a4 gis4 e2   %20
  a4 a4 a4 d4 
  cis4 b4 b2 
  gis4 a8 b8 cis4 b4 
  a2. a8 gis8 
  fis4 b4 a4 b4   %25
  a4 gis8 fis8 gis8 a8 b4 
  ais2. 
  
}

baixo = \relative c {
  \partial 4 fis8  e8 
  d4 cis8 b8 cis4 dis8 eis8 
  fis8 b,8 cis4 fis4 fis4 
  gis4 fis8 e8 a8 gis8 fis8 e8 
  d4 e4 a,4 a'8 gis8 
  fis4 gis8 fis8 eis4 fis8 e8   %5
  dis4 e4 b4 b'4 
  fis8 e8 d8 cis8 b4 cis4 
  d4 cis8 b8 fis'4 e4 
  a4 gis8 a8 b4 a8 b8 
  cis4 cis,4 fis4 fis8 gis8   %10
  a8 b8 cis4~ cis8 d8 cis8 b8 
  a8 eis8 fis8 cis8 d8 b8 e4 
  a,2 b2 
  cis4 d4 e4 e4 
  a,2. eis'4   %15
  fis4. fis8 gis8 ais8 b4 
  fis2 e4 e4 
  e4 e8 e8 e4 e4 
  e4 fis4 gis4 a4 
  d,4 e4 a,2   %20
  a8 b8 cis8 a8 d8 cis8 d8 e8 
  fis4 gis8 a8 b2 
  cis,4. b8 a4 b8 cis8 
  d2. a'4~ 
  a8 gis8 fis8 eis8 fis4 b,8 cis8   %25
  d8 cis8 b8 a8 b4 cis4 
  fis2. 
  
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