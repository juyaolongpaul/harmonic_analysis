\version "2.10.33"

\header {
  title = "242 - Wie bist du, Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \partial 4 b4 
  b4 a4 g2 
  fis4 g4 a4 g4 
  fis4 fis4 e4 fis4 
  g4 a4 b2 
  a4 c4 c4 b4   %5
  a4 a4 g4 a4 
  a4 a4 a4 fis4 
  g4 a4 b2 
  a4 fis4 g4 a4 
  b4 b4 a4 g8 fis8   %10
  fis2 e4
  
}

alto = \relative c' {
  \partial 4 e8  dis8 
  e8 b8 cis8 dis8 e8 dis8 e4 
  e8 dis8 e4. dis8 e4 
  e4 dis4 b4 dis4 
  e4 d8 c8 b8 d8 g4 
  fis4 e4 d4 d4   %5
  e4 d8 c8 b4 d4 
  d8 cis8 d8 e8 fis4 d4 
  d8 e8 fis4 g4 d4 
  d4 fis4 b,8 e8 d8 c8 
  b4 e4. dis8 e4   %10
  e4 dis4 b
  
}

tenor = \relative c' {
  \partial 4 g8  a8 
  b8 e4 a,8 b2 
  b4 b4 a8 b16 c16 b4 
  cis8 fis,8 b8 a8 g4 b4 
  b8 c8 b8 a8 g8 b8 d4 
  d4 c8 b8 a8 fis8 g4~   %5
  g4 fis4 g4 fis8 g8 
  a4 a4 d4 a4 
  b4 c4 d4 g,4~ 
  g8 fis8 b8 a8 g4 d'4~ 
  d8 f8 e8 d8 c4 cis4   %10
  b4. a8 gis4
  
}

baixo = \relative c {
  \partial 4 e8  fis8 
  g4. fis8 e8 fis8 g8 e8 
  b4 e4 fis4 g8 e8 
  ais,4 b4 e,4 b'4 
  e4 fis4 g4 g,8 b8 
  d4 a'8 g8 fis8 d8 g8 b,8   %5
  c8 a8 d4 g,4 d'8 e8 
  fis8 g8 fis8 e8 d8 e8 d8 c8 
  b8 c8 b8 a8 g8 a8 b8 c8 
  d4 dis4 e4 fis4 
  g4 gis4 a4 a4   %10
  b4 b,4 e
  
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