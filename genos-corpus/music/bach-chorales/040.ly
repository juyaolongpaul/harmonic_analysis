\version "2.10.33"

\header {
  title = "40 - Ach Gott und Herr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 c4 
    b4 a4 g4\fermata g4 
  a4 b4 c4\fermata d4 
  c4 b4 a4 b8 c8 
  a2 g4\fermata c4 
  b4 c4 d4\fermata d4   %5
  e4 d8 c8 d4\fermata g,4 
  a4 b4 c8 d8 e8 f8 
  d2 c4\fermata
  
}

alto = \relative c'' {
  \partial 4 g4 
    g4 fis4 d4 e8 d8 
  c4 d4 e4 g4~ 
  g8 fis8 g4 fis4 g4~ 
  g4 fis4 d4 g4 
  f4 g4 g4 g4   %5
  g4 fis4 g4 g4 
  d4 d4 c4. a'8 
  g4. f8 e4
  
}

tenor = \relative c' {
  \partial 4 e4 
    d4. c8 b4 c8 bes8 
  a8 g8 f4 g4 b4 
  c4 d4 d4 d4 
  e4 d8 c8 b4 c4 
  d4 e4 b4 b4   %5
  c4 c4 b4 c4 
  c4 b4 e8 f8 g8 c,8 
  c4 b4 g
  
}

baixo = \relative c {
  \partial 4 c4 
    g'4 d4 g,4\fermata c4 
  f8 e8 d4 c4\fermata g4 
  a4 b8 c8 d4 g4 
  c,4 d4 g,4\fermata e'4 
  d4 c4 g'4\fermata g4   %5
  c8 b8 a4 g4\fermata e4 
  fis4 gis4 a4 g8 f8 
  g4 g,4 c\fermata 
  
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