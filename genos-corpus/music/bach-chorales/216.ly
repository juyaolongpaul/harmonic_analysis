\version "2.10.33"

\header {
  title = "216 - Es ist genug, so nimm, Herr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  a2 b4 cis4 
  dis2 r4 dis4 
  e4 b4 b4 d4 
  cis2. b4 
  cis8 dis8 e4 e4 dis4   %5
  e2 a,2 
  b4 cis4 dis2 
  r4 dis4 e4 b4 
  b4 d4 cis2 
  r4 b4 cis8 dis8 e4   %10
  e4 dis4 e2 
  r4 b4 b4 b4 
  cis4 b4 d4 cis4 
  cis2 b2 
  r4 b4 b4 b4   %15
  cis4 b4 d4 cis4 
  cis2 b2 
  r4 e4 cis4 b4 
  a2 r4 e'4 
  cis4 b4 a2   %20
  
}

alto = \relative c' {
  e2 e4 d8 cis8 
  gis'2 r4 gis4 
  gis4. a8 gis8 fis8 gis8 e8 
  a2. gis4 
  a8 b8 cis4 b4 a16 gis16 a8   %5
  gis2 e2 
  e4 e4 fis2 
  r4 fis8 gis16 a16 gis8 fis8 gis8 a8 
  b4 gis4 a2 
  r4 fis4 g8 a8 b4   %10
  b8 a16 gis16 a8 fis8 gis2 
  r4 gis8 a8 b8 a8 b8 gis8 
  a4 b4 a4 a4 
  a8 gis8 a8 fis8 gis2 
  r4 fis4 b,4 b'4   %15
  ais4 a4 gis4 a4 
  a8 fis8 dis4 e2 
  r4 e4 e4 fis8 gis8 
  fis2 r4 b4 
  a4 gis4 e2   %20
  
}

tenor = \relative c' {
  cis2 b4 fis'4 
  fis2 r4 bis,4 
  cis8 dis8 e4 e4 e4 
  e2. e4 
  e8 fis8 g4 fis4 b,4   %5
  b2 cis2 
  b4 a4 a2 
  r4 b4 b4 e4 
  e4 e4 e2 
  r4 dis4 e8 fis8 g4   %10
  fis4 b,4 b2 
  r4 e4 e4 e4 
  e4 e4 fis4 e4 
  e4 dis4 e2 
  r4 dis4 e4 f4   %15
  e4 d4 e4 fis4 
  e8 a,8 fis4 gis2 
  r4 b4 a4 a8 gis8 
  cis2 r4 e4 
  e4 d16 cis16 d8 cis2   %20
  
}

baixo = \relative c' {
  a2 gis4 ais4 
  bis2 r4 gis4 
  cis8 b16 a16 gis8 fis8 e8 d8 cis8 b8 
  a2. e'4 
  a4 ais4 b4 b,4   %5
  e2 a2 
  gis4 g4 fis2 
  r4 b,4 e4. fis8 
  gis4 fis8 e8 a2 
  r4 a4 g8 fis8 e4   %10
  b'4 b,4 e2 
  r4 e8 fis8 gis8 fis8 gis8 e8 
  a8 b8 a8 gis8 fis8 gis8 a4  
  e2 e2 
  r4 a4 gis4 g4   %15
  fis4 f4 e4 dis4 
  e2 e2 
  r4 gis4 a8 cis,8 dis8 eis8 
  fis2 r4 gis4 
  a4 e4 a,2   %20
  
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