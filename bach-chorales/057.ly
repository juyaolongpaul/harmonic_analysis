\version "2.10.33"

\header {
  title = "57 - O Traurigkeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 e4 
    c8 b8 c4 b4 gis4 
  a4 a4 gis4 e'4 
  e4 e4 f4 e8 d8 
  d2 c2 
  b4 c4 d4 e4   %5
  d4 c4 b2 
  a4 b4 c4 d8 e8 
  b2 a2 
  
}

alto = \relative c'' {
  \partial 4 gis4 
    e4 e8 dis8 e4 e4 
  e4 d4 e4 gis4 
  a4 b4 a4 g4 
  a4 g8 f8 e2 
  g4 f8 e8 d8 d'8 c4   %5
  b4 a4 gis2 
  e4 e4 e4 a8 g8 
  f4 e8 d8 c2 
  
}

tenor = \relative c' {
  \partial 4 b4 
    a8 gis8 fis4 gis4 b4 
  c4 b8 a8 b4 b4 
  c4 b4 c8 d8 e8 c8 
  c4 b4 c2 
  d4 c4 g'4 g,8 a8   %5
  b8 e,8 e'4 e2 
  e8 d4 c4 b8 a4 
  a4 gis4 e2 
  
}

baixo = \relative c {
  \partial 4 e4 
    a4 a,4 e'4 d4 
  c4 f4 e4 e4 
  a4 gis4 a8 b8 c4 
  fis,4 g4 c,2 
  g4 a4 b4 c4   %5
  gis'4 a4 e2 
  fis4 gis4 a8 g8 f8 e8 
  d4 e4 a,2 
  
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