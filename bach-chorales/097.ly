\version "2.10.33"

\header {
  title = "97 - Nun bitten wir den heiligen Geist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    b4 b4 a4. gis8 
  fis4 e4 fis4 gis4 
  a2. cis8 d8 
  e4 fis4 e4. d8 
  cis8 b8 a8 gis8 fis4 gis4   %5
  a2 cis4 cis4 
  cis4 b4 cis2 
  a4 a4 b8 cis8 d4 
  cis8 b8 cis4 a2 
  b8 cis8 d4 cis8 b8 a8 gis8   %10
  fis4 e4 a8 b8 cis4 
  b4. a8 gis4 fis4 
  e2 fis4 gis4 
  a2 b2 
  a2.   %15
  
}

alto = \relative c' {
  \partial 4 e4 
    e4 e4 e4 e4 
  d4 e4 d8 fis8 e4 
  e2. a4 
  a4 a4 b8 a8 gis4 
  a4 e4 d8 cis8 b4   %5
  cis2 e8 fis8 gis4 
  gis8 fis8 eis8 fis8 eis8 fis4 eis8 
  fis4 fis4 gis8 fis8 e4 
  e2 d2 
  b'8 a8 gis4 a4 e4   %10
  d8 cis8 b4 a8 e'8 e4 
  fis2~ fis8 e4 dis8 
  b2 d4. d8 
  cis8 d8 e4 fis4 e8 d8 
  cis2.   %15
  
}

tenor = \relative c' {
  \partial 4 cis4 
    b8 a8 gis4 a8 b8 cis4 
  a4 a4 a8 d8 cis8 b8 
  cis2. e4 
  e4 d8 cis8 b4 e4 
  e8 d8 cis8 b8 a4 e4   %5
  e2 a4 gis4 
  a4 b4 b8 a8 gis8 cis8 
  cis4 d4 d8 cis8 b4 
  a8 g8 fis8 e8 fis2 
  e4 b'4 a2~   %10
  a4 gis8 fis8 e8 gis8 a8 gis8 
  fis4 b4 b4 cis8 b8 
  gis2 b4 b4 
  a2. gis4 
  e2.   %15
  
}

baixo = \relative c' {
  \partial 4 a4 
    gis8 fis8 e8 d8 cis8 b8 a4 
  d4 cis4 b4 e4 
  a,2. a'8 b8 
  cis4 d4 gis,4 e4 
  a4 cis,4 d4 e4   %5
  a,2 a'4 eis4 
  fis4 d4 cis2 
  fis4 d8 cis8 b8 a8 gis4 
  a2 d2 
  gis4 fis8 e8 a4 cis,4   %10
  d2. cis4 
  dis8 b8 cis8 dis8 e8 gis,8 a8 b8 
  e2 b'4 eis,4 
  fis4 cis4 d4 e4 
  a,2.   %15
  
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