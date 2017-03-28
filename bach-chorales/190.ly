\version "2.10.33"

\header {
  title = "190 - Herr, nun laß in Frieden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  e4 e4 fis4 gis4 
  a2 gis2 
  gis4 gis4 a4 b4 
  c2 b2 
  c4 c4 b4 b4   %5
  a2 gis2 
  a4 a4 g4 g4 
  f2 e2 
  e'4 e4 d4 d4 
  c2 b2   %10
  a4 a4 g4 g4 
  f2 e2 
  
}

alto = \relative c' {
  b4 c4 d4 d4 
  e4 d4 e2 
  e4 e4 e8 fis8 gis4 
  a2 a4 gis4 
  g8 fis8 g8 a8 g4 f4   %5
  e4 fis4 e2 
  c4 f8 e8 d8 c8 d8 e8 
  d2 c2 
  g'4 g4 g4 g8 f8 
  e2 e2   %10
  c4 f4 f4 e4~ 
  e4 d8 c8 b2 
  
}

tenor = \relative c' {
  g4 a8 gis8 a4 b4 
  c4 b8 a8 b2 
  b4 b4 c4 d4 
  e2 e2 
  e4. d8 d4 d4~   %5
  d8 c4 b8 b2 
  a8 g8 f4 g4 d8 a'8 
  a4 g4 g2 
  c4 c2 b4~ 
  b4 a2 gis4   %10
  a8 b8 c4 c4 c8 bes8 
  a8 c8 b8 a8 gis2 
  
}

baixo = \relative c {
  e8 d8 c4. d8 c8 b8 
  a8 g8 f4 e2 
  d'8 f8 e8 d8 c8 d8 c8 b8 
  a8 b8 c8 d8 e2 
  e8 d8 e8 fis8 g8 fis8 g8 gis8   %5
  a4 dis,4 e2 
  f8 e8 d8 c8 b8 a8 b8 cis8 
  d8 c8 b8 g8 c2 
  c8 d8 e8 f8 g8 a8 b8 g8 
  a,8 b8 c8 d8 e2   %10
  f,8 g8 a8 b8 c8 d8 e8 c8 
  d2 e2 
  
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