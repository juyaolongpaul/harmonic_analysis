\version "2.10.33"

\header {
  title = "370 - Kommt her zu mir, spricht Gottes Sohn"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 a4 e'4 d4 
  e4 c4 b4 a8 b8 
  c4 b4 c8 d8 e4 
  d4 c4 b4 c8 b8 
  a4 a4 d8 c8 b8 c8   %5
  a2 g4 g4 
  c4 b4 c8 d8 e4 
  d4 c4 b4 a8 b8 
  c4 b4 c8 d8 e4 
  d4 c4 b4 e4   %10
  e4 e4 d8 c8 b8 c8 
  b8 gis8 a8 b8 b2 
  a2. 
  
}

alto = \relative c' {
  \partial 4 e8  f8 
    e4 e8 d8 e4 f4 
  e4 e4 e4 e4 
  a4 gis4 a4 g8 a8 
  b4 a4 gis4 a4 
  d,8 e8 fis4 fis4 g4   %5
  g4 fis4 d4 e8 f8 
  g4 d4 e8 f8 g4 
  gis4 a4 gis4 e4 
  d4 d4 e8 f8 g4 
  f8 e8 e4 e4 gis4   %10
  a4 g4 f4. e8 
  f4 e8 f8 e8 f8 e8 d8 
  cis2. 
  
}

tenor = \relative c' {
  \partial 4 c8  d8 
    c8 b8 a4 a8 c8 b8 a8 
  b4 a4 gis4 a4 
  e'4 e4 e4 e4 
  e4 e4 e4 e4 
  a,4 d4 d4 e4   %5
  d8 c16 b16 c4 b4 c4 
  g8 a8 b8 a8 g4 c4 
  d4 e4 e4 c8 b8 
  a4 b8 a8 g4 c4 
  c8 b8 b8 a8 gis4 b4   %10
  c4 cis4 d4. e8 
  d4 c8 a8 gis8 a4 gis8 
  e2. 
  
}

baixo = \relative c {
  \partial 4 a4 
    a'4 f4 c4 d4 
  gis,4 a4 e'4 c8 b8 
  a4 e'4 a8 b8 c4 
  gis4 a4 e4 a8 g8 
  fis8 e8 d8 c8 b4 e8 c8   %5
  d2 g,4 c8 d8 
  e8 f8 g8 f8 e4 d8 c8 
  b4 a4 e'4 a8 g8 
  fis8 d8 g8 f8 e8 d8 c8 b8 
  a8 gis8 a4 e'4 e4   %10
  a8 g8 f8 e8 f8 e8 d8 c8 
  d8 b8 c8 d8 e8 d8 e4 
  a,2. 
  
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