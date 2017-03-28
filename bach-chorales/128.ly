\version "2.10.33"

\header {
  title = "128 - Alles ist an Gottes Segen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  g4 g4 d'4 d4 
  c4 d4 b4 g4 
  b4 b4 a4 a4 
  g4 a4 fis4 d4 
  a'4 a4 b8 a8 g8 fis8   %5
  e4. d8 d2 
  d'4 d4 d4 e4 
  c4 d4 b4 g4 
  b4 b4 a4 a4 
  g4 a4 fis4 d4   %10
  g4 a4 b8 c8 d8 c8 
  b4 a8 g8 g2 
  
}

alto = \relative c' {
  d4 d4 d8 a'4 g8 
  a4 a8 fis8 g4 d4 
  g4 g4 g4 fis4~ 
  fis8 e16 d16 e8 cis8 d4 a4 
  d4 d4 d4 d4~   %5
  d8 b8 cis4 a2 
  a'4 a4 gis8 a8 b4~ 
  b8 a16 g16 a4 g4 d4 
  g4 g8 fis8 e8 cis8 fis4~ 
  fis8 e16 d16 e8 cis8 d4 a4   %10
  d8 b8 fis'4 g8 fis8 g4~ 
  g4 fis4 d2 
  
}

tenor = \relative c' {
  b4 b4 a4 b4 
  e4 d4 d4 b4 
  d4 d4 d4 d4 
  d8 b8 a4 a4 fis4 
  fis8 g8 a8 fis8 g4 a4   %5
  a4. g16 fis16 fis2 
  fis'4 fis4 f4 e4 
  e4 d4 d4 b4 
  d8 b8 e8 d8 cis8 a8 d4 
  d8 b8 a4 a4 fis4   %10
  g4 d'4 d8 c8 b8 c8 
  d4 d8. c16 b2 
  
}

baixo = \relative c' {
  g4 g4 fis4 g4~ 
  g4 fis8 d8 g4 g,4 
  g8 a8 b8 c8 d8 e8 fis8 d8 
  b8 e8 cis8 a8 d4 d,4 
  d'8 e8 fis8 d8 g4 d4   %5
  a'4 a,4 d2 
  d8 d'4 c8 b8 a8 gis8 e8 
  a8 g8 fis8 d8 g4 g,4 
  g'4 g4 g4 fis8 d8 
  b8 e8 cis8 a8 d4 d,4   %10
  b'8 e8 d8 c8 b8 a8 g8 e'8 
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