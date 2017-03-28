\version "2.10.33"

\header {
  title = "136 - Herr Jesu Christ, dich zu uns wend"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \partial 4 g4 
    b4 d4 b4 a4 
  b4 cis4 d4 e4 
  d8 c8 b4 a4 g8 a8 
  b4 a4 g4 a4 
  b4 b4 a4 d4   %5
  d4 cis4 d4 b4 
  c4 b4 a4 g4 
  g4 fis4 g2 
  
}

alto = \relative c' {
  \partial 4 d4 
    g4 a4. g4 fis8 
  g4 g4 fis4 a4 
  a4. g4 fis8 e8 d8 
  d4 d4 d4 fis4 
  g4 g4 g4 fis4   %5
  e4 e4 fis4 d4 
  e4 d8 g4 fis4 e8 
  d4 d4 d2 
  
}

tenor = \relative c' {
  \partial 4 b4 
    d4 d4 d4 d4 
  d4 e4 a,4 c4 
  d4 d4 e8 b4 a8~ 
  a8 g4 fis8 b4 d4 
  d4 d4 d4 a8 d8   %5
  b8 g8 a4 a4 g4 
  g8 a8 b4 e8 b8 b4 
  b4 a4 b2 
  
}

baixo = \relative c {
  \partial 4 g4 
    g4 fis4 g8 b8 d4 
  g8 fis8 e4 d4 a'8 g8 
  fis4 g4 cis,8 dis8 e8 fis8 
  g4 d4 g,4 d'4 
  g,8 a8 b8 cis8 d8 e8 fis8 b8   %5
  g8 e8 a8 a,8 d4 g8 f8 
  e8 fis8 g4 cis,8 dis8 e4 
  b8 c8 d4 g,2 
  
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