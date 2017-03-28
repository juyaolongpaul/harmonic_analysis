\version "2.10.33"

\header {
  title = "169 - Jesu, der du selbsten wohl"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  cis4 a4 b4 b4 
  cis4 dis4 e2 
  b4 cis4 d4 cis4 
  b2 a2 
  b4 b4 e4 e4   %5
  d4 d4 cis2 
  b4 b4 a4 gis4 
  fis2 e2 
  a4 a4 d4 d4 
  cis4 cis4 b2   %10
  a4 a4 a4 a4 
  b2 e,2 
  e'4 d4 cis4 b4 
  a4 gis4 fis2 
  b4 a4 d4 cis4   %15
  b2 a2 
  
}

alto = \relative c' {
  e4 e4 fis4 e4 
  e4 fis4 gis2 
  gis4 a4. gis8 a4 
  a4 gis4 e2 
  gis4 gis4 gis4 ais4   %5
  fis4 fis4 fis2 
  fis4. e8 e8 dis8 e4 
  e4 dis4 b2 
  e4 e4 fis8 gis8 a8 b8~ 
  b8 a16 gis16 a4 a4 gis4   %10
  fis4 g4 fis4 fis4 
  fis8 e8 fis4 b,2 
  cis8 a'4 gis8 a4. gis8~ 
  gis8 fis4 eis8 fis2 
  fis8 gis8 a4 a8 gis8 a4   %15
  a4 gis4 e2 
  
}

tenor = \relative c' {
  a4 a4 a4 gis4 
  a4 a4 b2 
  e4 e4 d8 b8 e4 
  fis8 d8 b8 e16 d16 cis2 
  e4 e4 e8 d8 cis4~   %5
  cis4 b4 b4 ais4 
  dis8 cis8 b4 cis8 a8 b4 
  cis4 b8 a8 gis2 
  a8 b8 cis8 b8 a8 gis8 fis8 e8 
  e'4 e4 e2   %10
  cis8 d8 e2 d8 cis8 
  b8 gis8 a8 b16 a16 gis2 
  a8 cis8 d8 b8 e8 fis8 b,8 e16 d16 
  cis4~ cis8. b16 a2 
  d4 e4 d8 b8 cis8 e8   %15
  fis8 d8 b8 e16 d16 cis2 
  
}

baixo = \relative c {
  a4 cis4 d4 e4 
  a8 gis8 fis4 e2 
  e8 d8 cis4 b4 a4 
  d4 e4 a,2 
  e'4 e8 d8 cis4 fis4   %5
  b,8 cis8 d8 e8 fis2 
  b8 a8 gis4 fis4 e4 
  a,4 b4 e2 
  cis8 b8 a8 gis8 fis8 e8 fis8 gis8 
  a8 b8 cis8 d8 e2   %10
  fis8 e8 d8 cis8 d8 e8 fis8 e8 
  dis8 e4 dis8 e2 
  a,4 b4 cis8 d8 e4 
  fis4 cis4 d2 
  d8 e8 d8 cis8 b8 e8 a,8 cis8   %15
  d8 b8 e4 a,2 
  
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