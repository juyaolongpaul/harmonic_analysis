\version "2.10.33"

\header {
  title = "159 - Als der gütige Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c' {
  \partial 4 d4 
    g4 a4 b4 a4 
  g2 r4 a4 
  b4 c4 b4 a4 
  g2 r4 b4 
  d4 d4 c4 b4   %5
  a2 r4 b4 
  g4 c4 b4 a4 
  g2 a4 a4 
  b4 c4 b4 a4 
  g2.   %10
  
}

alto = \relative c' {
  \partial 4 b4 
    e8 g4 fis8 g8 fis16 e16 fis4 
  d2 r4 fis4 
  g4 g8 a4 g4 fis8 
  d2 r4 g4 
  g4 g4. fis8 g4~   %5
  g4 fis4 r4 fis4~ 
  fis8 e8 g8 a4 g4 fis8 
  g4. fis16 e16 fis4 fis4 
  g4 g4 g4. fis8 
  d2.   %10
  
}

tenor = \relative c' {
  \partial 4 g4 
    b4 d4 d4 d8 c8 
  b2 r4 d4 
  d4 e4 d4. d16 c16 
  b2 r4 d4 
  d4 b4 c4 d4   %5
  d2 r4 b4 
  b4 e8 d8 d4. d16 c16 
  b8 a8 b8 cis8 d4 d4 
  d4 c8 e8 d4 d8. c16 
  b2.  %10
  
}

baixo = \relative c' {
  \partial 4 g4 
    e4 d4 g4 d4 
  g,2 r4 d'4 
  g4 c,4 d4 d,4 
  g2 r4 g4 
  b4 g4 a4 b8 c8   %5
  d2 r4 dis4 
  e4. fis8 g4 d4 
  e2 d4 d4 
  g4 e8 c8 d4 d,4 
  g2.  %10
  
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