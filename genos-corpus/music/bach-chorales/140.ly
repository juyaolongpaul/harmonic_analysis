\version "2.10.33"

\header {
  title = "140 - In allen meinen Taten"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 g4 
    c4 c4 d4 d4 
  e2 d4 d4 
  c4 b4 a4 a4 
  b2 b4 g4 
  g4 g4 c4 c4   %5
  b2 r4 b4 
  e4 e4 c4 c4 
  d2 b4 b4 
  c4 c4 a4 a4 
  b2 g4 g4   %10
  c4 d4 e4 d4 
  c2. 
  
}

alto = \relative c' {
  \partial 4 e4 
    e8 f8 g4 a4 g4 
  g2 g4 g4~ 
  g8 fis8 g4 g4 fis4 
  g2 g4 d4 
  e4 e4 a4 a4   %5
  gis2 r4 g4 
  g4 g4 f4 f4 
  fis8 g8 a8 fis8 g4 g4 
  g4 g4 fis4 fis4 
  fis4 b,4 b4 b4   %10
  c4 f4 e8 f8 g8 f8 
  e2. 
  
}

tenor = \relative c' {
  \partial 4 c4 
    c4 c4 c4 b4 
  c2 b4 b4 
  c4 d4 e4 d4 
  d2 d4 b4 
  c4 c8 d8 e4 e4   %5
  e2 r4 e4~ 
  e8 d8 c8 b8 a8 b8 c8 a8 
  a4 d4 d4 d4 
  e4 e4 e4 d4 
  dis8 e8 fis8 dis8 e4 d4   %10
  g,4 b4 c4 b4 
  g2. 
  
}

baixo = \relative c {
  \partial 4 c4 
    c8 d8 e4 f4 g4 
  c,8 d8 e8 f8 g4 g,4 
  a4 b4 c4 d4 
  g2 g,4 g'8 f8 
  e8 d8 c8 b8 a8 b8 c8 d8   %5
  e2 r4 e8 d8 
  c8 d8 e8 c8 f8 g8 a8 f8 
  d8 e8 fis8 d8 g4 g,4 
  c8 d8 e8 c8 d8 e8 fis8 d8 
  b8 cis8 dis8 b8 e4 g8 f8   %10
  e8 f8 e8 d8 c4 g'4 
  c,2. 
  
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