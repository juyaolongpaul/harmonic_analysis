\version "2.10.33"

\header {
  title = "213 - O wie selig seid ihr doch, ihr Frommen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  a4 d4 c4 bes8 a8 
  bes4 a4 g4 a8 bes8 
  g2 f2 
  a8 bes8 c4 b4. cis8 
  d4 c8 d8 e4 d8 c8   %5
  b2 a2 
  r4 a4 g4 e4 
  f2 e2 
  d4 e4 f4 g4 
  a4 b4 cis4 d4   %10
  d4 cis4 d2 
  
}

alto = \relative c' {
  f4 g4 g4 f4~ 
  f8 e8 f4. e8 f4 
  f4 e4 c2 
  f4 g8 a8 g4 g4 
  a8 b8 a4 a4 a4   %5
  a4 gis4 e2 
  r4 cis4 d4 e4 
  e4 d4 cis2 
  d4 a4 d4 c4 
  c4 f4 e4 d8 e8   %10
  f8 e16 d16 e4 fis2 
  
}

tenor = \relative c' {
  d4 d4 e4 d4 
  d8 bes8 c4 c4 c4 
  d8 bes8 g8 c8 a2 
  c8 d8 e8 d8 d4 e4 
  d8 b8 e4 e4 f4   %5
  b,8 e4 d8 cis2 
  r4 e,4 d4 a'4 
  a2 a2 
  a4 a4 a8 bes8 a8 g8 
  f8 a8 g8 f8 g4 a8 bes8   %10
  a4 a4 a2 
  
}

baixo = \relative c {
  d4 c8 bes8 a4 d4 
  g,4 a8 bes8 c4 f4 
  bes,4 c4 f,2 
  f'4 e8 fis8 g8 f8 e4 
  fis8 gis8 a4 c,4 d4   %5
  e4 e,4 a2 
  r4 a4 b4 cis4 
  d4 d,4 a'2 
  f'4 cis4 d4 e4 
  f4 e8 d8 e4 f8 g8   %10
  a4 a,4 d2 
  
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