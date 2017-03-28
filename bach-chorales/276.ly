\version "2.10.33"

\header {
  title = "276 - Lobt Gott, ihr Christen, allzugleich"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 g4 
  d'4 d4 d4 d4 
  e4 d8 c8 b4 a4 
  b8 cis8 d4 d4 cis4 
  d2. d4 
  d4 d4 d8 c8 b8 c8   %5
  d8 c8 b4 a4 d4 
  c4 b4 a4 a4 
  g8 a8 b8 c8 d4 d4 
  c4 b4 a4 a4 
  g2.   %10
  
}

alto = \relative c' {
  \partial 4 d4 
  d8 e8 fis4 g4 g8 f8 
  e8 g8 fis4 g4 a4 
  g4 fis4 e4 e4 
  d2. fis4 
  g4 fis4 g8 a8 b8 a8   %5
  g4 g8 fis16 e16 fis4 fis4 
  g4 g8 fis8 e4 fis4 
  g4 fis4 f4 e4 
  e8 fis8 g8 fis8 e4 fis4 
  d2.   %10
  
}

tenor = \relative c' {
  \partial 4 b4 
  a4 a8 b16 c16 b8 a8 g4 
  g4 d'4 d4 d4 
  g,4 d'8 cis8 b8 e8 a8 g8 
  fis2. d8 c8 
  b8 c8 d8 c8 b8 c8 d4   %5
  d4 e4 a,4 b8 a8 
  g8 a8 b8 d8 e4 d4 
  d4 c4 b4 b4 
  c8 d8 e4 e8 c8 a8 b16 c16 
  b2.  %10
  
}

baixo = \relative c' {
  \partial 4 g4 
  fis4 d4 g8 a8 b8 g8 
  c4 d4 g,4 fis4 
  e4 b'8 a8 g4 a4 
  d,2. d4 
  g8 a8 b8 a8 g4. a8   %5
  b4 c4 d4 b4 
  e,4. d8 c8 a8 d8 c'8 
  b4 a4 gis4 gis,4 
  a4 e'8 d8 c8 a8 d4 
  g,2.   %10
  
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