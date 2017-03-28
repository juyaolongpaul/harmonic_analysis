\version "2.10.33"

\header {
  title = "19 - Ich hab' mein' Sach Gott heimgestellt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \partial 4 g4 
    g4 g4 fis4 bes4 
  a4 g4 fis4\fermata fis4 
  fis4 fis4 g4 e4 
  fis4 fis4 g4\fermata a4 
  a4 c4 a4 f4   %5
  g4 a4 bes4\fermata bes4 
  a4 g4 fis4\fermata fis4 
  fis4 fis4 g4 e4 
  fis4 fis4 g4\fermata 
  
}

alto = \relative c' {
  \partial 4 d4 
    d4 d8 cis8 d4 g8 f8 
  ees8 d8 e4 d4 d4 
  c4 d4 d4 c4 
  c8 ees8 d4 d4 f4 
  f4 g4 f4 f4   %5
  ees8 d8 c8 ees8 d4 g4~ 
  g8 fis8 g8 g,8 d'4 d4 
  c8 d8 ees8 d8 d4 e4 
  d4 d4 d4 
  
}

tenor = \relative c' {
  \partial 4 bes4 
    bes8 a8 g4 a4 d4 
  c4 bes8 a8 a4 a4 
  a4 a4 g4 g4 
  a8 c8 c8 bes16 a16 bes4 c4 
  c4 c4 c4 bes4   %5
  bes4 a8 c8 bes4 d4 
  d4. c8 a4 a4 
  a4 a4 g4~ g16 a16 bes8 
  a8 g8 a16 bes16 c8 c8 b8 
  
}

baixo = \relative c' {
  \partial 4 g4 
    g8 f8 ees4 d8 c8 bes8 g8 
  c4 cis4 d4\fermata d8 ees8~ 
  ees8 d4 c4 b8 c8 bes8 
  a4 d4 g,4\fermata f'4 
  f8 g8 f8 e8 f8 ees4 d8   %5
  ees8 e8 f8 fis8 g4\fermata g,4 
  d'4 ees4 d4\fermata d4 
  a8 bes8 c4 bes8 b8 c8 cis8 
  d4 d,4 g4\fermata 
  
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