\version "2.10.33"

\header {
  title = "217 - Ach Gott, wie manches Herzeleid"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key c \major
}


soprano = \relative c'' {
  c4 c4 c4 
  a4 b4 c4 
  d4 c4 b4 
  a2. 
  a4 b4 c4   %5
  d2 c4 
  b4 a2 
  g2. 
  e'4 e4 e4 
  c4 d4 e4   %10
  f4 e2 
  d2. 
  c4 d4 e4 
  d2 c4 
  c4 b2   %15
  c2. 
  
}

alto = \relative c' {
  e4 g4 g4 
  f4 g4 g4 
  a2 g4 
  fis2. 
  fis4 g4 g4   %5
  g2 g4 
  g4 g4 fis4 
  g2. 
  g4 e4 gis4 
  a4 b4 cis4   %10
  d4 c2 
  g2. 
  g4 b4 c4 
  c4 b4 e,4 
  a4 g2   %15
  g2. 
  
}

tenor = \relative c' {
  g4 c4 e4 
  c4 d4 e4 
  d2 d4 
  d2. 
  d4 d4 e4   %5
  d2 e4 
  d4 d4. c8 
  b2. 
  c4 b4 e4 
  e4 f4 g4   %10
  f4 g4 c,4 
  b2. 
  c4 f,4 g4 
  a4 b4 c4 
  d4 d2   %15
  e2. 
  
}

baixo = \relative c {
  c4 e4 c4 
  f2 e4 
  fis2 g4 
  d2. 
  d4 g4 e4   %5
  b2 c4 
  g'4 d2 
  g,2. 
  c'4 gis4 e4 
  a2 a,4   %10
  d4 e4 f4 
  g2. 
  e4 d4 c4 
  f4 g4 a4 
  fis4 g4 g,4   %15
  c2. 
  
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