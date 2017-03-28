\version "2.10.33"

\header {
  title = "170 - Nun komm, der Heiden Heiland"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  a4 a4 g4 c4 
  b8 a8 b4 a2 
  a8 b8 c4 d4 c4 
  d4 e4 c2 
  c4 d4 e8 d8 c4   %5
  d8 c8 b4 a2 
  a4 a4 g4 c4 
  b8 a8 b4 a2 
  
}

alto = \relative c' {
  e4 d4 e4 e4 
  e8 fis8 gis4 e2 
  e8 gis8 a4 d,4 g4 
  f4 e8 f16 g16 f2 
  f4 f4 e4 e4   %5
  d8 a'8 gis4 e2 
  dis8 e8 fis4 e4 e4 
  f8 c8 d4 cis2 
  
}

tenor = \relative c' {
  c4 b8 a8 b8 a8 g8 a8 
  b8 c8 d4 c2 
  c8 d8 e8 f8 b,8 a8 g8 a8 
  b4 c8 g8 a2 
  a4 b8 c8 b4 a4   %5
  a4 e'8 d8 c2 
  fis,4 b4 b4 a4 
  a4. gis8 e2 
  
}

baixo = \relative c' {
  a8 g8 f4 e4. fis8 
  gis8 a8 e4 a,2 
  a4 a'4 g8 f8 e4 
  d4 c4 f2 
  f8 e8 d4 gis4 a8 g8   %5
  f8 d8 e4 a,2 
  b8 cis8 dis8 b8 e8 d8 c8 a8 
  d8 e16 f16 e4 a,2 
  
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