\version "2.10.33"

\header {
  title = "160 - Gelobet seist du, Jesu Christ"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 g4 
    g4 g4 a4 g4 
  c4 d4 c2 
  b4 d4 e4 d4 
  b4 a4 g4 g4 
  c4 b4 d4 a4   %5
  g4 e4 d4 d4 
  a'4 a4 a4 b4 
  c4 a4 g8 a8 b4 
  a2 g4 g4 
  g1   %10
  
}

alto = \relative c' {
  \partial 4 d4 
    e4 e4 f4 g4 
  a4 b4 a2 
  g4 g4 g4 fis4 
  g4 fis4 d4 e4 
  e8 fis8 g4 g4 f4   %5
  e8 d8 cis4 a4 d4 
  d4 a'8 g8 fis4 gis4 
  a4 f4 e8 fis8 g4~ 
  g4 f4. e8 e4~ 
  e4 d8 c8 d2   %10
  
}

tenor = \relative c' {
  \partial 4 b4 
    b4 c4 c4 c8 d8 
  e4 f4 e2 
  e4 d4 c8 b8 a4 
  b8 c8 d4 b4 b4 
  c4 d4 d4 d8 c8   %5
  b4 a8 g8 fis4 fis8 g8 
  a8 b8 cis4 d4 d4 
  e4 c4 c4 d4~ 
  d8 a8 d8 c8 b4 c4~ 
  c4 b8 a8 b2   %10
  
}

baixo = \relative c' {
  \partial 4 g4 
    e4 c4 f4 e4 
  a4 gis4 a2 
  e4 b'4 c4 d4 
  g,4 d4 g4 e4 
  a4 g4 b,8 c8 d4   %5
  g,4 a4 d4 d8 e8 
  fis8 g8 a8 b8 c4 b4 
  a4 a,8 b8 c4 b8 c8 
  d4 c8 d8 e4 d8 c8 
  g'1   %10
  
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