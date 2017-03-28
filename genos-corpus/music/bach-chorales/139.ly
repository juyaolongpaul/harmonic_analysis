\version "2.10.33"

\header {
  title = "139 - Warum sollt ich mich denn grämen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  g4 g4 a4 g4 
  a8 b8 c4 c4 b4 
  c2 d4 e4 
  c2 b8 c8 d4 
  a2 d4 d4   %5
  c4 b8 a8 a2 
  g2 g4 g4 
  a4 g4 a8 b8 c4 
  c4 b4 c2 
  d4 e4 c2   %10
  b8 c8 d4 a2 
  d4 d4 c4 b8 a8 
  a2 g2 
  
}

alto = \relative c' {
  d4 e4 d4 d4 
  e4 f4 g2 
  g2 b8 a8 b4 
  a2 g4 g4 
  fis2 fis4 g4   %5
  g4 g4 g4 fis4 
  d2 e4 b8 cis8 
  d8 c8 b4 c8 d8 c8 d8 
  e4 d4 c2 
  fis4 gis4 e2   %10
  g4 fis8 g8 a2 
  g4 g8 fis8 g8 a8 d,8 e8 
  fis8 g4 fis8 d2 
  
}

tenor = \relative c' {
  b4 c4 fis,4 g4 
  c4 c8 d8 e4 d4 
  e2 g8 f8 e4 
  e2 e4 d4 
  d2 a4 b4   %5
  c4 d4 e4 d8 c8 
  b2 c4 g4 
  fis4 g4 g8 f8 g8 a8 
  g4 e'4 e2 
  d8 c8 b4 a2   %10
  g8 a8 b8 g8 d'2 
  b8 c8 d4 e8 fis8 g4 
  d8 c16 b16 c4 b2 
  
}

baixo = \relative c' {
  g4 c,2 b4 
  a4 aes4 g2 
  c2 g'4 gis4 
  a2 e4 b8 c8 
  d2 d4 g8 fis8   %5
  e4 d4 c4 d4 
  g,2 c8 d8 e4 
  d8 e8 f4 e8 d8 e8 fis8 
  g4 gis4 a2 
  b4 e,4 a2   %10
  e4 d8 e8 fis2 
  g8 a8 b4. a8 b8 c8 
  d4 d,4 g2 
  
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