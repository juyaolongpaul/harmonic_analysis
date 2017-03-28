\version "2.10.33"

\header {
  title = "79 - Heut' triumphieret Gottes Sohn"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key a \minor
}


soprano = \relative c'' {
  c4 c4 d4 
  e2 d4 
  c2 b4 
  c2. 
  c4 c4 c4   %5
  d4 c4 b4 
  a2 gis4 
  a2 b4 
  c2 d4 
  e4. f8 g4   %10
  f4( e4) d4 
  e2. 
  e4 e4 e4 
  d2 d4 
  e4 d4 c4   %15
  b2. 
  b4 b4 cis4 
  d4 c4 b4 
  a2 gis4 
  a2 b4   %20
  c2 d4 
  e2 e4 
  e4 dis2 
  e2. 
  
}

alto = \relative c' {
  e8 f8 g4 g4 
  g2 f4 
  e4 d2 
  e2. 
  f4 g4 f4   %5
  f4 e4 d4 
  e4 f4 e4 
  e2 e4 
  e2 g4 
  g2 g4   %10
  c4 b4 a4 
  gis2. 
  a4 g8 f8 g8 a8 
  b4 g4 a4 
  b2 a4   %15
  gis2. 
  g8 a8 g8 f8 e4 
  d4 e4 f4 
  e4 f4 e4 
  e2 gis4   %20
  a2 g4 
  g2 fis4 
  fis4 gis4 a4 
  gis2. 
  
}

tenor = \relative c' {
  c4 c4 b4 
  c4 b4 a4 
  g4 a4 g4 
  g2. 
  a4 g4 a4   %5
  gis4 a4 b4 
  c4 b2 
  c2 gis4 
  a8 b8 c4 b4 
  c2 d4   %10
  c4 e4 f4 
  b,2. 
  c4 c4 c4 
  d8 c8 d8 e8 f4 
  e2 e4   %15
  e2. 
  e4 e4 a,4 
  a2 gis4 
  a4 d4 b4 
  c2 e4   %20
  e4 f4 d4 
  c2 c4 
  b2 b4 
  b2. 
  
}

baixo = \relative c' {
  a4 e4 g4 
  c,2 d4 
  e4 f4 g4 
  c,2. 
  f4 e4 f4   %5
  b,4 c4 d4 
  c4 d4 e4 
  a,2 e'4 
  a2 g4 
  c2 b4   %10
  a4 g4 f4 
  e2. 
  a4 e4 c4 
  g'4 b4 a4 
  gis2 a4   %15
  e2. 
  e8 d8 e8 f8 g4 
  f4 e4 d4 
  c4 d4 e4 
  a,2 e'4   %20
  a2 b4 
  c4 b4 a4 
  b2 b,4 
  e2. 
  
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