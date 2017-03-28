\version "2.10.33"

\header {
  title = "181 - Gott hat das Evangelium"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \partial 4 b4 
    b4 b8 c8 d8 c8 b4 
  a4 b8 a8 g4 b4 
  b4 b8 c8 d8 c8 b4 
  a4 b4 g4 b4 
  c4 b4 a4 g8 a8   %5
  b4 fis4 e4 b'4 
  c4 b4 a4 g8 a8 
  b4 fis4 e4 b'4 
  b4 a4 g8 fis8 e8 fis8 
  g4 g4 a4 a4   %10
  b2.
  
}

alto = \relative c'' {
  \partial 4 g4 
    fis4 g4 a4. g8~ 
  g8 fis16 e16 fis4 d4 g4 
  a4 g4 fis4 g4~ 
  g4 fis4 e4 g4~ 
  g8 a4 g4 fis8 e4~   %5
  e8 dis16 cis16 dis4 e4 gis4 
  a4 d,8 e8 fis4 e4 
  e4 dis4 e4 g4 
  fis8 e8 fis8 dis8 b2 
  e4 e4 e8 g8 fis8 e8   %10
  fis2. 
  
}

tenor = \relative c' {
  \partial 4 e4 
    b4 e4 d4 d4 
  e4 d8 c8 b4 d8 e8 
  fis4 e4 d4 e4 
  e4 b4 b4 e4~ 
  e8 d8 d4 e8 b8 b4   %5
  b4. a8 g4 d'4 
  e8 fis8 g4 d8 c8 b4~ 
  b4. a8 g4 e'4 
  b4 fis4 e8 fis8 g8 a8 
  b4 b4 c8 e8 dis8 e8   %10
  dis2.
  
}

baixo = \relative c {
  \partial 4 e4 
    dis4 e4 fis4 g4 
  c,4 d4 g,4 g'4 
  dis4 e4 b4 e8 d8 
  cis4 dis4 e4 e4~ 
  e8 fis8 g4 cis,8 dis8 e8 fis8   %5
  g8 a8 b8 b,8 c4 b4 
  a4 b8 c8 d4 e4 
  g,8 a8 b4 e,4 e'4 
  dis8 cis8 dis8 b8 e2~ 
  e8 fis8 g8 e8 c8 b8 c4   %10
  b2.
  
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