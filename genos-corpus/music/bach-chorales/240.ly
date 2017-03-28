\version "2.10.33"

\header {
  title = "240 - Nun sich der Tag geendet hat"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 e4 
  a,4 b4 c4 c4 
  d4 d4 e4 e4 
  d4 e4 c4 c4 
  b2 r4 e4 
  c4 b4 c4 a4   %5
  gis4 a4 b4 e4 
  d4 c4 b4. b8 
  a2. 
  
}

alto = \relative c'' {
  \partial 4 a4 
  e4 e4 e8 f8 g4 
  a4 g4 g4 g4 
  a4 e4 e4 e4 
  e2 r4 b'4~ 
  b8 a8 gis4 a8 g8 f4~   %5
  f8 e8 e8 fis8 gis4 a4~ 
  a8 gis8 a4 a4 gis4 
  e2. 
  
}

tenor = \relative c' {
  \partial 4 c4 
  c4 b4. a8 g8 c8~ 
  c8 b16 a16 b4 c4 c4~ 
  c4 b2 a4~ 
  a4 gis4 r4 e'4 
  e4 e4 e4 d4~   %5
  d8 b8 c4 b4 a4 
  d4 e4 f8 d8 b8 e16 d16 
  cis2.
  
}

baixo = \relative c {
  \partial 4 a8  b8 
  c8 b8 a8 gis8 a4 e4 
  f4 g4 c,4 c'4 
  fis,4 gis4 a8 b8 c8 d8 
  e2 r4 gis,4 
  a4 e'8 d8 c4 d8 c8   %5
  b8 e8 c8 a8 e'4 c4 
  b4 a4 d4 e4 
  a,2.
  
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