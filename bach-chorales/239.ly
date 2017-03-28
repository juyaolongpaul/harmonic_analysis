\version "2.10.33"

\header {
  title = "239 - Den Vater dort oben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  g4 g4 a4 b4 
  c2 c2 
  e4 d4 c4 b4 
  a2 g2 
  g4 g4 a4 b4   %5
  c4 g4 g2 
  a4 g4 f4 e4 
  d4 d4 c2 
  c'4 d4 c4 b4 
  a2 g2   %10
  g4 g4 a4 b4 
  c4 g4 g2 
  a4 g4 f4 e4 
  d2 c2 
  
}

alto = \relative c' {
  e4 d4 e8 fis8 g4 
  g4 f2 e4 
  g4 fis8 g16 fis16 e8 a8 d,8 c16 d16 
  e8 c8 a8 d16 c16 b2 
  e4 e4 e8 fis8 g4   %5
  g8 f4 e16 d16 e2 
  f8 e8 d8 e16 d16 c8 b8 c4~ 
  c4 c8 b8 g2 
  g'4 g4. fis8 g4 
  g4 fis4 g2   %10
  e8 f8 g4. f16 e16 f4 
  f8 e8 d4 c2 
  c8 f4 e4 d8 c4~ 
  c8 b16 a16 b4 g2 
  
}

tenor = \relative c' {
  c4 g4 c4 d4 
  c8 bes8 a4 g2 
  c8 b8 a8 b16 a16 g8 fis8 g4~ 
  g4. fis8 d2 
  b'4 b4 c4 d4   %5
  c4 d8 g,8 c2 
  c4 b8 c16 b16 a8 d8 g,8 f16 g16 
  a8 f8 d8 g16 f16 e2 
  e'4 d4 g,8 a8 b8 c16 d16 
  e8 c8 a8 d16 c16 b2   %10
  c4 c4 c4 d4 
  g,4 g8. f16 e2 
  f4 g8 c16 bes16 a4 a4 
  a8 d,8 d8 g16 f16 e2 
  
}

baixo = \relative c {
  c4. b8 a4 g8 f8 
  e4 f4 c2 
  c'4. b4 a8 g8 a16 b16 
  c8 a8 d4 g,2 
  e8 e'4 d8 c4 b4   %5
  a4 b4 c2 
  f4. e4 d8 c8 d16 e16 
  f8 d8 g8 g,8 c2 
  c4. b8 e4 d4 
  c4 d4 g,2   %10
  c8 d8 e4 f8 e8 d4 
  c4. b8 c2 
  f4 c4 d4 a8 g8 
  f4 g4 c,2 
  
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