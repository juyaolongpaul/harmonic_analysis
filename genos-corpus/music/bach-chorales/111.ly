\version "2.10.33"

\header {
  title = "111 - Herzliebster Jesu, was hast du verbrochen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 a4 gis4 e4 
  a4 b4 c4 c4 
  d4 c16 b16 c8 b4 b4 
  c4 d8 e16 f16 e4 c4 
  f4 f4 e8 d8 c8 d8   %5
  d2 c4 c4 
  b4 a4 g8 fis8 e8 fis8 
  g4 g4 a4 g4 
  f2 e4 e'4 
  d4 c8 b8 b2   %10
  a1 
  
}

alto = \relative c' {
  \partial 4 c8  d8 
    e4 e4 e4 b4 
  e8 fis8 gis4 a4 a4 
  a2 gis4 gis4 
  a4 g4 g4 f4 
  a4 a4 a4 e4   %5
  a4 g4 g4 a4 
  a8 g8 g8 fis8 e4 e4 
  e8 d8 cis8 d8 e4 e4 
  e4 d4 cis4 a'4 
  a8 b8 e,8 d8 f4 e8 d8   %10
  cis1 
  
}

tenor = \relative c' {
  \partial 4 a8  b8 
    c4 c4 b4 gis4 
  a4 d4 e4 e4 
  f4 a,4 e'4 e4 
  e4 b4 c4 a4 
  d4 d4 c4 c4   %5
  c4 b4 e4 e4 
  fis8 e8 c4 b4 b4 
  b4 ais4 a4 a4 
  a2 a4 c4 
  c8 b8 a4 a4 gis4   %10
  e1 
  
}

baixo = \relative c' {
  \partial 4 a4 
    a,8 b8 c8 d8 e8 f8 e8 d8 
  c8 d8 c8 b8 a8 e'8 a8 g8 
  f8 e8 f8 d8 e4 e4 
  a8 g8 f8 g8 c,8 d8 e8 f8 
  d8 e8 f8 g8 a8 b8 a8 g8   %5
  f8 e8 f8 g8 c,4 a'4 
  dis,8 e8 fis8 dis8 e8 fis8 g8 fis8 
  e8 f8 e8 d8 cis8 a8 b8 cis8 
  d8 e8 f8 g8 a4 a8 g8 
  fis8 gis8 a8 f8 d8 b8 e4   %10
  a,1 
  
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