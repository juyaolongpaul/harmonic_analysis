\version "2.10.33"

\header {
  title = "202 - O wir armen Sünder"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  a4 a4 a4 a4 
  b2 a2 
  g4 fis4 e4 e4 
  d1 
  a'4 a4 a4 a4   %5
  b2 g2 
  b4 b4 c4 b4 
  a1 
  a4 a4 a4 a4 
  d2 a4 a4   %10
  b4 b4 a4 a4 
  g1 
  a4 a4 a4 a4 
  a2 fis2 
  g4 fis4 e4. d8   %15
  d1 
  a'2 b2 
  c1 
  a2 gis2 
  a1   %20
  d2. e4 
  a,2 b2 
  cis1 
  a2. gis4 
  a1   %25
  a4. g8 fis4 g4 
  e2 d2 
  
}

alto = \relative c' {
  fis4 g4 fis8 g8 a4~ 
  a8 g16 fis16 g2 fis4 
  e4. d4 cis16 b16 cis4 
  a1 
  d4 cis8 d8 e8 cis8 d8 e8   %5
  fis8 g8 fis4 e2 
  g4 gis4 a4. gis8 
  e1 
  fis4 fis8 e8 fis8 g8 a4~ 
  a4 g4 fis4 fis4   %10
  e8 dis8 e4 e4 d8 c8 
  b1 
  e8 cis8 d8 e8 fis4 fis4 
  e8 g8 fis8 e8 e8 dis16 cis16 dis4 
  e4. d8 d8 cis16 b16 cis4   %15
  a1 
  fis'2 f4 e4 
  e4 f8 g8 a4 g4~ 
  g4 f4 e2 
  e1   %20
  b'2. a8 g8 
  fis2~ fis8 gis8 a4 
  gis2. cis,4~ 
  cis4 d8 e8 fis4 e8 d8 
  e1   %25
  cis8 d8 e4. d8 d4~ 
  d8 cis16 b16 cis4 a2 
  
}

tenor = \relative c' {
  d4 e4 e4 d4 
  d2 d2~ 
  d8 cis16 b16 a4 b4 a8 g8 
  fis1 
  fis4 e4 a8 g8 fis4~   %5
  fis8 e8 b'4 b2 
  e4 e4 e8 a,8 f'8 e8 
  cis1 
  d4 d8 cis8 d8 e8 fis8 e8 
  d8 c8 d4 d4 d8 c8   %10
  b8 a8 g4. fis16 e16 fis4 
  g1 
  e8 g8 fis8 e8 d8 a'8 d4~
  d8 cis8 c4 b2 
  b4 b4 b8 g8 e8 a8   %15
  fis1 
  a4 d2 c8 b8 
  a4 g4 f4 e4 
  d4 d'2 c8 b8 
  cis1   %20
  fis,4 e8 fis8 g4 e4 
  fis8 gis8 a2 fis4 
  cis'4 b8 a8 gis8 fis8 gis4 
  a4 b8 cis8 d2~ 
  d4 d4 cis4 b4   %25
  a2~ a16 b16 c4 b16 a16 
  b8 e,4 fis16 g16 fis2 
  
}

baixo = \relative c {
  d4. cis8 d8 e8 fis8 d8 
  g4 g,4 d'2 
  e4 fis4 g4 a8 a,8 
  d1 
  d4 a8 b8 cis8 a8 d8 cis8   %5
  dis8 e4 dis8 e2 
  e8 d8 c8 b8 a8 f8 d8 e8 
  a1 
  d8 e8 fis8 g8 fis8 e8 d8 c8 
  b8 a8 b8 cis8 d4 d4   %10
  g8 fis8 e8 d8 c8 cis8 d8 dis8 
  e1 
  cis8 a8 b8 cis8 d8 e8 fis8 g8 
  a4 a,4 b2 
  e,8 g8 b8 a8 g8 e8 a4   %15
  d,1 
  d'2 gis,2 
  a2. b8 c8 
  d4 b4 e2 
  a,1   %20
  b2. cis4 
  d2 dis2 
  e2 eis2 
  fis2 b,2 
  a1   %25
  fis'4 cis4 d4 g,4 
  gis4 a4 d,2 
  
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