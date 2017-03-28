\version "2.10.33"

\header {
  title = "228 - Danket dem Herren, denn er ist sehr freundlich"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 a4 
  c4 b4 c4 d4 
  e4 e4 d4 c4 
  b2 b4 b4 
  e4 d4 d4 c4 
  b4 c4 d4 c4   %5
  b4 b a
  
}

alto = \relative c' {
  \partial 4 e4 
  a4 b4. a4 g8 
  g4 g4 g8 f8 e4 
  e2 e4 e4 
  e4 fis4 gis4 a4~ 
  a8 gis8 a4 a8 gis8 a4   %5
  a4. gis8 e4
  
}

tenor = \relative c' {
  \partial 4 c4 
  e4 e4 e4 d4~ 
  d8 c16 b16 c4 b4. a8~ 
  a4 gis8 fis8 gis4 gis4 
  a4 a4 b4 c8 d8 
  e4 e4 d4 e4   %5
  f4 e8. d16 c4
  
}

baixo = \relative c {
  \partial 4 a4 
  a'4. gis8 a8 a,8 b4 
  c8 d8 e8 f8 g4 a4 
  e2 e,4 e'8 d8 
  c4 c4 b4 a4 
  e'4 a8 g8 f4 e4   %5
  d4 e a,
  
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