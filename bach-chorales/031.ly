\version "2.10.33"

\header {
  title = "31 - Ach lieben Christen, seid getrost"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    c4 a8 b8 c8 d8 e4 
    d8 e16 f16 d4 c4 c4 
    b4 a8 b8 c4 d8 c8 
    b2 a4 } d4 
  e4 e4 b8 c8 d4   %5
  c4 c4 b4 c4 
  d4 a8 b8 c4 b4 
  a16 b16 c8 a4 g4 c4 
  b4 a8 b8 c4 d8 c8 
  b2 a4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e8  f8 
    g4 f4 g4 g4 
    a4 g8 f8 e4 fis4 
    g4 f4 e4 a4 
    a4 gis4 e4 } g4 
  g4 g4 d8 e8 f4   %5
  e4 a4 gis4 a4 
  a8 g8 fis4 e4 d8 g8 
  g8 e8 fis4 d4 e4 
  d8 e8 fis8 f8 e4 a4~ 
  a4 gis4 e   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    c4 c4 c4 c4 
    c4 b4 g4 a4 
    d4 d4 c8 b8 a4 
    f'4 e8 d8 cis4 } b4 
  c4 g8 a8 b4 a8 b8   %5
  c8 d8 e4 e4 e4 
  a,4 d4 g,8 a8 b8 d8 
  e4 d8 c8 b4 g8 a8 
  b8 c8 d4 c8 b8 a8 c8 
  f8 d8 b8 e8 cis4   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    e4 f4 e8 d8 c4 
    f4 g4 c,4 a'4 
    g4 d'4 a8 g8 f8 e8 
    d4 e4 a,4 } g4 
  c8 d8 e8 f8 g4 f4   %5
  a,8 b8 c8 d8 e4 a8 g8 
  fis8 e8 d4 e8 fis8 g4 
  c,4 d4 g,4 e'8 fis8 
  g4 fis8 gis8 a8 g8 f8 e8 
  d8 b8 e4 a,   %10
  
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