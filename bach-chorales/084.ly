\version "2.10.33"

\header {
  title = "84 - Nun bitten wir den heiligen Geist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    b4 b4 a4 gis8 fis8 
  e4 fis8 gis8 a4 b4 
  a2. cis8 d8 
  e4 fis4 e4. d8 
  cis8 b8 a8 gis8 fis4 gis4   %5
  a2 cis4 cis4 
  cis4 b4 cis2 
  a4 a4 b8 cis8 d4 
  cis4. b8 a2 
  b8 cis8 d4 cis8 b8 a8 gis8   %10
  fis4 gis8 a8 b4. a8 
  gis4 fis4 e2 
  fis4 gis4 a2~ 
  a4 gis a
  
}

alto = \relative c' {
  \partial 4 e4 
    e4 e4 e4 d8 cis8 
  b8 cis8 a8 d8 cis4 fis8 e8 
  e2. a4 
  b8 cis8 d8 cis8 b8 a8 gis8 fis8 
  gis8 eis8 cis4 d4 d4   %5
  cis2 e4 e4 
  e8 fis8 gis8 fis8 eis8 fis4 eis8 
  fis4 fis4 gis8 a8 b4 
  b4 a8 g8 fis2 
  fis8 gis8 a8 b8 a8 gis8 fis8 e8   %10
  dis4 e8 fis8 gis4. fis8~ 
  fis8 e4 dis8 b2 
  d4 e4 e8 cis8 fis4 
  e2 e4
  
}

tenor = \relative c' {
  \partial 4 cis4 
    b8 a8 gis4 a4 b8 a8 
  gis8 a8 fis8 b8 e,8 a4 gis8 
  cis2. e4 
  e4 a,4 b8 cis8 d4 
  gis,4 a4 a4 b4   %5
  e,2 a4 a4 
  gis8 a8 b4 b8 a8 gis8 cis8 
  cis4 cis4 b8 a4 gis16 fis16 
  e4 a4 a2 
  d8 cis4 b8 cis8 dis8 e4   %10
  b4 b4 b8 cis16 dis16 e8 b8 
  b4. a8 gis2 
  a4 b8 d8 cis4 c4 
  b8 a8 b8 d8 cis4
  
}

baixo = \relative c' {
  \partial 4 a4 
    gis8 fis8 e8 d8 cis8 a8 d4~ 
  d8 cis8 d8 b8 cis8 fis8 d8 e8 
  a,2. a'4 
  a8 gis8 fis4 gis8 a8 b4 
  eis,8 cis8 fis8 e8 d8 cis8 b4   %5
  a2 a8 b8 cis8 d8 
  e4. d8 cis2 
  fis4 fis8 e8 d8 cis8 b8 e8 
  a,8 b8 cis4 d2 
  d8 e8 fis8 gis8 a8 b8 cis8 a8   %10
  b8 a8 gis8 fis8 e8 dis8 cis8 dis8 
  e8 gis8 b8 b,8 e2 
  d8 cis8 d8 b8 cis8 fis8 e8 dis8 
  e4 e, a
  
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