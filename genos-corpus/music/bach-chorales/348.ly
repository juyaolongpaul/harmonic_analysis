\version "2.10.33"

\header {
  title = "348 - Meinen Jesum laß ich nicht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  g4 g4 a4 a4 
  b4 b4 c2 
  d4 d4 c4 c4 
  b4 a8 b16 c16 a2 
  g2 g4 g4   %5
  a4 a4 g4 f4 
  e2 g4 g4 
  f4 f4 e4 d8 e16 f16 
  d2 c2 
  c'4 c4 b4 b4   %10
  a4 a4 g2 
  g4 g4 f4 e4 
  d4 d4 c2 
  
}

alto = \relative c' {
  e4 e4 f4 f4 
  f8 e8 d4 e2 
  g4 g4 g4 fis4 
  d4 c8 d16 e16 fis2 
  d2 e8 d8 c4   %5
  c4 d4 d8 e8 f8 d8 
  c2 b4 e4 
  d4 d4 c8 bes8 a4 
  g2 g2 
  e'8 d8 c4 g'2~   %10
  g8 a16 g16 fis16 e16 fis8 b,2 
  e4 e4 d4 c8 b8 
  a4 b4 g2 
  
}

tenor = \relative c' {
  c4 c4 c4 d4 
  d4 g,4 g2 
  d'4 b4 a4 a4 
  g8 fis8 e8 c'8 c8 b8 a4 
  b2 c4 g4   %5
  f4 f4 g4 c8 g8 
  g2 g4 a4 
  a4 g4 g4. f8~ 
  f8 e8 d4 e2 
  g4. fis8 d'4 d4   %10
  d4 c8 a8 e'2 
  cis4 a4 a2~ 
  a8 d,8 g8 f8 e2 
  
}

baixo = \relative c' {
  c8 bes8 a8 g8 f8 e8 d4 
  g4 g,4 c2 
  b8 c8 d8 e8 a,8 b8 c8 d8 
  g,4 c4 d2 
  g,2 c8 d8 e8 c8   %5
  f8 e8 d8 c8 b8 c8 a8 b8 
  c2 e8 d8 cis8 a8 
  d8 c8 b8 g8 c4 f,4 
  g2 c2 
  c8 b8 a4 g8 a8 b8 c8   %10
  d4 dis4 e2 
  a,8 b8 cis8 a8 d4 a8 g8 
  f4 g4 c2 
  
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