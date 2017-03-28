\version "2.10.33"

\header {
  title = "219 - O wie selig seid ihr doch, ihr Frommen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  d4 a4 c4 d8 e8 
  f4 e4 d4 c4 
  bes2 a2 
  f'4 d4 c4 a8 bes8 
  c4. c8 a4 bes4   %5
  g2 f4 a4 
  g4 f4 e2 
  d2 a'4 c4 
  d4 a4 c4 a4 
  g4 f4 e2   %10
  d1 
  
}

alto = \relative c' {
  f8 g8 a4 e4 f8 g8 
  a4 a8. g16 f4 f4 
  d8 bes8 c4 c2 
  f4 f8 d8 g8 e8 c4 
  f4 e4 f8 a8 g8 f8~   %5
  f8 e16 d16 e4 c4 f4~ 
  f8 e4 d4 cis16 b16 cis4 
  d2 c8 d8 e4 
  f8 g8 a8 f8 g8 e8 f4 
  e4 d4 d4 cis4   %10
  a1 
  
}

tenor = \relative c' {
  a4 d4 c8 bes8 a4 
  d4. c4 bes4 a8~ 
  a8 g8 f8 e8 f2 
  c'8 a8 d8 bes8 g4 f8 g8 
  a4 g4 f8 c'8 d4   %5
  c8 g8 c8 bes8 a4 c4 
  c8. bes16 a4 bes8 e,8 a8 g8 
  f2 f4 g4 
  a4 d4 c4 c4 
  bes4 a8 bes8 bes8 g8 e8 a16 g16   %10
  fis1 
  
}

baixo = \relative c {
  d8 e8 f8 g8 a8 g8 f8 e8 
  d4 a'4 bes4 f4 
  g4 c,4 f2 
  a8 f8 bes4 e,8 c8 f4~ 
  f8 a,8 bes8 c8 d8 c8 bes4   %5
  c4 c,4 f4 f'4 
  c4 d4 g,4 a4 
  bes2 f'4 e4 
  d8 e8 f8 d8 e8 c8 f8 e8 
  d8 cis8 d8 bes8 g8 e8 a4   %10
  d,1 
  
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