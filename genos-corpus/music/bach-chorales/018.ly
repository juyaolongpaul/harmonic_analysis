\version "2.10.33"

\header {
  title = "18 - Gottes Sohn ist kommen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  g4 g8 a8 b4 c4 
  d4 e4 d2\fermata 
  e4 fis4 g4 d4 
  b4 cis4 d2\fermata 
  c4 a4 b4 c4   %5
  b4 a4 g2\fermata 
  fis4 g4 a4 b4 
  a4. g8 fis4 e4 
  d1\fermata 
  g4 g8 a8 b4 c4   %10
  d4 e4 d2\fermata 
  g,4 a4 b4 c4 
  b4 a4 g2\fermata 
  
}

alto = \relative c' {
  d4 e8 fis8 g4 fis8 e8 
  d8 fis8 b8 a8 fis2 
  a4 a4 g4 a4 
  g8 fis8 e4 fis2 
  e4 d4 d4 c4   %5
  d4. c8 b2 
  d4 cis8 b8 a4 d4 
  e2~ e8 d4 cis8 
  a1 
  d4 e8 fis8 g4 g4   %10
  f4 e8 fis8 g2 
  g8 e8 d8 c8 b8 a8 g8 fis'8 
  g4. fis8 d2 
  
}

tenor = \relative c' {
  b4 c4 d4 c4 
  b16 c16 d4 cis8 d2 
  c8 b8 c8 d8 b4 a4 
  d4 a4 a2 
  g4 a4 g8 fis8 g4   %5
  g4 fis4 g2 
  a4 g4 d'8 cis8 b4 
  e,4 a2 g4 
  fis1 
  g4 c4 d4 c4~   %10
  c8 b8 c4 b2 
  b4 a4 d4 e8 a,8 
  b8 c8 d4 b2 
  
}

baixo = \relative c' {
  g4 c,4 g8 g'8 a4 
  b8 a8 g8 a8 d,2\fermata 
  a'4 d,4 e4 fis4 
  g4 a4 d,2\fermata 
  e4 fis4 g4 e4   %5
  d8 c8 d4 g,2\fermata 
  d'4 e4 fis4 g4 
  cis,4 a4 d4 a4 
  d1\fermata 
  b4 c4 g'4 e4   %10
  d4 c4 g'2\fermata 
  e4 fis4 g4 e4 
  d2 g2\fermata 
  
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