\version "2.10.33"

\header {
  title = "357 - Warum sollt' ich mich denn grämen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  g4 g4 a4 g4 
  a8 b8 c4 c4 b4 
  c2 d4 e4 
  c2 b8 c8 d4 
  a2 d4 d4   %5
  c4 b4 a4 a8. g16 
  g2 c4 c4 
  b4 c4 a4 d4 
  d4 cis4 d2 
  d4 e4 c2   %10
  b8 c8 d4 a2 
  d4 d4 c4 b8 a8 
  a4 a8. g16 g2 
  
}

alto = \relative c' {
  e8 f8 g8 e4 d4 g8 
  f4 g4 g2 
  g2 g8 f8 e4 
  e2 e4 d4 
  d2 d4 e4   %5
  e8 d4 g8 g4 fis4 
  d2 g8 fis8 g8 a8 
  g4 g2 f4 
  e2 fis2 
  g4 g4 f2   %10
  d8 e8 d8 e8 fis2 
  fis4 g8 fis8 e8 fis8 g4 
  g4 fis4 d2 
  
}

tenor = \relative c' {
  c4 c4 c4 b8 c8 
  f4 e4 d8 e8 f8 g16 f16 
  e2 b4 b4 
  a2 g4 g4 
  fis2 fis4 g4~   %5
  g8 a8 b8 d8 e4 a,16 b16 c8 
  b2 g4 e'4 
  d4 c4 c4 a4 
  bes4 a4 a2 
  b4 bes8 c16 bes16 a2   %10
  g4 g4 d'2 
  d8 c8 b4 c4 d4 
  d4 a16 b16 c8 b2 
  
}

baixo = \relative c {
  c8 d8 e8 c8 f4. e8 
  d4 e8 f8 g2 
  c,2 g'4 gis4 
  a2 e4 b8 c8 
  d2 b4 e4~   %5
  e8 fis8 g8 b,8 c8 a8 d4 
  g,2 e'8 d8 e8 fis8 
  g8 f8 e8 c8 f8 e8 f8 d8 
  g8 e8 a8 a,8 d2 
  g4 c,4 f2   %10
  f8 e8 b8 c8 d2 
  b4 e4 a,4 b8 c8 
  d2 g,2 
  
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