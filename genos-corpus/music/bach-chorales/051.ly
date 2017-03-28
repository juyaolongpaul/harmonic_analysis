\version "2.10.33"

\header {
  title = "51 - Gelobet seist du, Jesu Christ"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 g4 
    g4 g4 a4 g4 
  c4 d4 c2 
  b8 c8 d4 e4 d8 c8 
  b4 a4 g4 g4 
  c4 b8 c8 d4 a4   %5
  g8 fis8 e4 d4 d4 
  a'4 a4 a4 b4 
  c8 b8 a4 g8 a8 b4 
  a2 g4 g4 
  g2 g4   %10
  
}

alto = \relative c' {
  \partial 4 d4 
    e4 d8 e8 fis4 e4 
  g4 g8 f8 e2 
  g8 a8 b4 c8 b8 a4 
  g4 fis4 d4 d8 e8 
  fis4 g4 g4 fis8 e8   %5
  d8 b8 cis4 a4 a4 
  d4 cis4 fis4 g4 
  a8 g4 fis8 e8 a8 g4 
  fis2 d8 e8 f4 
  e4 e4 d   %10
  
}

tenor = \relative c' {
  \partial 4 b4 
    c4 b4 a8 d8 b4 
  e8 c8 b4 g2 
  e'4 f4 g4 d4 
  d4. c8 b4 g4 
  a4 d8 c8 b8 g8 d'4   %5
  d4 a8 g8 fis4 fis8 g8 
  a8 b8 cis8 e8 d4 d4 
  e8 d8 e8 d4 c8 b8 c8 
  d2 d4 b4 
  g4 c4 b   %10
  
}

baixo = \relative c' {
  \partial 4 g8  fis8 
    e8 fis8 g4 d4 e8 d8 
  c4 g4 c2 
  e4 d4 c4 fis,4 
  g8 b8 d4 g,4 b4 
  a4 g8 a8 b8 c8 d8 c8   %5
  b8 g8 a4 d4 d8 e8 
  fis8 g8 a8 cis,8 d8 c8 b4 
  a8 b8 c8 d8 e8 fis8 g4 
  d2 b8 a8 g4 
  c8 e16 d16 c8 c,8 g'4   %10
  
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