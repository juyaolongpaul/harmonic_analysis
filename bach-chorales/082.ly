\version "2.10.33"

\header {
  title = "82 - O großer Gott von Macht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \partial 4 a4 
  bes4 c4 d4 a8 b8 
  c2 r4 g4 
  g4 a4 bes4 c4 
  a2 r4 f4 
  f4 g4 a4 bes4   %5
  c2 r4 d4 
  bes8 c8 d4 c4 c4 
  bes2 r4 d4 
  d4 d4 c4 c4 
  bes4 bes4 a2   %10
  r4 a4 bes4 a4 
  g4 g4 f4 f4 
  e2 r4 g4 
  g4 a8 bes8 c4 bes4 
  a2 g2   %15
  r4 g4 g4 a8 b8 
  c4 bes4 a2 
  a1 
  
}

alto = \relative c' {
  \partial 4 fis4 
  g4 f?4 f4 f4 
  g2 r4 ees4 
  e?4 fis4 g4 a4 
  fis2 r4 d4 
  d4 e4 f4 ees4   %5
  ees2 r4 d4 
  g8 a8 bes4 bes4 a4 
  f2 r4 f4 
  g4 g4 g4 d4 
  d4 e4 fis2   %10
  r4 fis4 g4 f4 
  f4 e4 d4 d4 
  c2 r4 d4~
  d8 c8 f4 g8 a8 d,8 g8 
  g4 fis4 d2   %15
  r4 d4 ees4 f4 
  g4 g4 g8 f8 e4 
  fis1 
  
}

tenor = \relative c' {
  \partial 4 d4 
  d4 c4 bes4 f'4 
  ees2 r4 c4 
  c4 c4 d4 ees4 
  d2 r4 a4 
  bes4 bes4 c4 bes4   %5
  f'2 r4 f4 
  ees4 f4 f4 c4 
  d2 r4 bes4 
  bes4 bes4 bes4 a4~
  a4 g4 a2   %10
  r4 d4 d4 c4 
  d4 g,4 a4 g4 
  g2 r4 g4 
  g4 d'4 c8 a8 bes8 g8 
  ees'4 a,4 bes2   %15
  r4 b4 c4 f4 
  ees4 d4 cis8 d4 cis8 
  d1 
  
}

baixo = \relative c {
  \partial 4 d4 
  g4 a4 bes4 d4 
  c2 r4 c4 
  bes4 a4 g4 c,4 
  d2 r4 d8 c8 
  bes8 bes'8 a8 g8 f4 g4   %5
  a2 r4 bes4 
  ees,4 d8 ees8 f4 f4 
  bes,2 r4 bes'8 a8 
  g4 f4 e4 fis4 
  g4 g4 d2   %10
  r4 d4 g,4 a4 
  b4 c2 b4 
  c2 r4 bes4 
  ees4 d4 e8 fis8 g4 
  c,4 d4 g,2   %15
  r4 f'4 ees4 d4 
  c4 g4 a2 
  d1 
  
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