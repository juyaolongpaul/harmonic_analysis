\version "2.10.33"

\header {
  title = "211 - Weltlich Ehr und zeitlich Gut"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  c4 b4 a4 a4 
  a4 b4 c8 b8 a4 
  g2 r4 b4 
  c4 b4 c4 a4 
  b4 g4 a2   %5
  b2 r4 b4 
  b4 b4 a4 b4 
  c2 g4 g4 
  a4 b4 c4 a4 
  b2 b4 c4   %10
  d4 d4 c4 a4 
  b2 d4 d4 
  d4 d4 e4 d8 c8 
  b2 b4 g4 
  a4 b8 c8 d8 c8 b4   %15
  a4 b4 c2 
  
}

alto = \relative c' {
  e8 f8 g4. f16 e16 f4 
  f8 a8 g4 g4. f8 
  f4 e4 r4 g4 
  g4 g4 g4 f4~ 
  f4 e4 e2   %5
  e2 r4 e4 
  fis8 a4 g8 fis4 e4 
  e2 d8 f4 e8~ 
  e8 d8 d4 c8 g'4 fis8 
  g2 g4. fis8   %10
  g8 a8 b4 a8 g8 fis4 
  g2 a4 g8 a8 
  b8 a8 g8 f8 e4 fis4 
  g2 g8 f8 e4~ 
  e8 fis8 g4 a4. g8~   %15
  g8 f16 e16 f8 e16 d16 e2 
  
}

tenor = \relative c' {
  g4 g8 c8 c4 c4 
  d8 f8 e8 d8 c4 c4 
  c2 r4 d4 
  e4 d4 c4 c4 
  d4 b4 a8 b8 c8 a8   %5
  gis2 r4 g4 
  b8 fis'4 e4 d8 c8 b8 
  a2 d4. c16 b16 
  c4 b8 a8 g8 e'8 d4 
  d2 d4 c4   %10
  b4 e4 e4 d4 
  d2 fis8 e8 d8 c8 
  b4 b4 c8 b8 a4 
  e'2 d4 c4 
  c4 d4 d4 d4   %15
  d4 d8 g,8 g2 
  
}

baixo = \relative c {
  c8 d8 e4 f8 g8 f8 e8 
  d4 g8 f8 e4 f4 
  c2 r4 g'4 
  c,8 e8 g8 f8 e8 c8 f8 e8 
  d8 b8 e8 d8 c8 b8 a8 c8   %5
  e2 r4 e4 
  dis4 e4 fis4 gis4 
  a2 b4 c4 
  fis,4 g8 fis8 e8 c8 d4 
  g,2 g4 a4   %10
  b4 gis4 a8 b8 c8 d8 
  g,2 d''8 c8 b8 a8 
  g8 f8 e8 d8 c4 d4 
  e2 g,4 c8 b8 
  a4 g4 fis4 g4   %15
  d'4 g,4 c2 
  
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