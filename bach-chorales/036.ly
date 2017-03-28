\version "2.10.33"

\header {
  title = "36 - Nun bitten wir den heiligen Geist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    b4 b4 a4 fis4 
  e4 fis8 gis8 a4 b4 
  a2. cis8 d8 
  e4 fis4 e4 cis4 
  a4 fis8 gis8 a4 b4   %5
  a2 cis4 cis4 
  cis4 b4 cis2 
  a2 r4 a4 
  b4 b4 cis2 
  a2 b4 b4   %10
  cis8 b8 a8 gis8 fis4 b4 
  b4 a8 gis8 fis2 
  e2 fis4 gis4 
  a4 b4 a
  
}

alto = \relative c' {
  \partial 4 e4 
    e4 e4 e4 d8 cis8 
  b4 cis8 d8 e4 e4 
  e2. a4 
  b8 cis8 d8 cis8 b4 a4 
  a4 d,4 cis4 fis8 e8   %5
  cis2 a'4 b4 
  a8 gis8 fis4 fis4 eis4 
  cis2 r4 fis4 
  gis4 gis4 a8 fis8 g4 
  fis2 fis4 gis4   %10
  a8 gis8 fis8 e8 d4 d4 
  cis8 gis'8 fis8 e8 e4 dis4 
  b2 d4 d4 
  e4 e8 d8 cis4
  
}

tenor = \relative c' {
  \partial 4 cis4 
    b8 a8 gis4 a4 a4 
  gis4 a8 b8 e,8 a4 gis8 
  cis2. e4~ 
  e4 a,4 b4 cis8 d8 
  e4 a,4 a4. gis8   %5
  e2 e'4 gis4 
  fis4 fis,8 gis8 a4 gis4 
  fis2 r4 cis'4 
  e4 e4 e2 
  d2 d8 cis8 b4   %10
  a8 cis8 d8 e8 a,4 gis8 fis8 
  e4 fis8 gis8 cis4 b8 a8 
  gis2 a4 b4 
  e,8 a4 gis8 e4 
  
}

baixo = \relative c' {
  \partial 4 a4 
    gis8 fis8 e8 d8 cis4 d4~ 
  d4 cis8 b8 cis8 a8 e'4 
  a,2. a'4 
  gis4 fis4 gis4 a4 
  cis,4 d8 e8 fis8 a8 d,8 e8   %5
  a,2 a'8 gis8 fis8 eis8 
  fis8 e8 d4 cis2 
  fis2 r4 fis4 
  e8 d8 cis8 b8 a2 
  d2 b4 e4   %10
  a,4 b8 cis8 d4 b4 
  cis4 dis8 e8 a,4 b4 
  e2 d4 cis8 b8 
  cis8 a8 e4 a
  
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