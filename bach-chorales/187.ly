\version "2.10.33"

\header {
  title = "187 - Komm, Gott Schöpfer, heiliger Geist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 g4 
    a4 g4 f4 g4 
  c4 d4 c4 c4 
  g4 b4 c4 d4 
  e4 e4 d4 d4 
  e8 d8 c4 a4 g4   %5
  b4 cis4 d4 g,4 
  c4 a4 f4 a4 
  g2 g4
  
}

alto = \relative c' {
  \partial 4 d8  e8 
    f4~ f8 e~ e d16 c16 b8 e8~ 
  e8 c8 g'8 f8 e4 c8 d8 
  e4 f4 g4 g4 
  g4~ g8 fis g4 g4 
  g4 g4~ g8 f~ f e8   %5
  g4 g4 a4 e4 
  e8 d8 e4~ e8 d16 e16 f4~ 
  f4 e4 d
  
}

tenor = \relative c' {
  \partial 4 b4 
    c4 c4 a4 g8 c16 b16 
  c8 f8 b,4 g4 a4 
  b8 c8 d4 e4 d4~ 
  d8 c16 b16 c8 a8 b4 b4 
  c8 b8 c4 c4 c4   %5
  d4 e4 f4 b,4 
  a4 a4 a8 ais8 c4 
  d4~ d8 c~ c b
  
}

baixo = \relative c' {
  \partial 4 g4 
    f4 c4 d4 e4 
  a,4 g4 c4 f4 
  e4 d4 c4. b8 
  c4 c,4 g'4 g'4 
  c,4 e4 f4 c'4   %5
  g8 f8 e4 d4 e4 
  a,8 b8 c4 d4 a4 
  b4 c4 g
  
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