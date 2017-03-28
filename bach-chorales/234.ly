\version "2.10.33"

\header {
  title = "234 - Gott lebet noch"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key f \major
}


soprano = \relative c' {
  f4 a4 b4 
  c2. 
  c4 bes4 c4 
  d4 c4 d4 
  e4 d4 e4   %5
  f2. 
  a,2 bes4 
  c4 bes4 a4 
  g2 a4 
  bes4 a4 g4   %10
  f2 g4 
  a2 g4 
  a2 b4 
  c2. 
  e2 f4   %15
  d2 f4 
  e2 d4 
  cis2 a4 
  d2 a4 
  bes4 a4 g4   %20
  f2 e8 f8 
  d2. 
  fis2 fis4 
  g4 fis4 g4 
  a4 bes4 c4   %25
  bes4 a8 bes8 g4 
  c2 c4 
  c4 d4 e4 
  f4 e4 d4 
  e2 c4   %30
  f4 c4 a4 
  d2 c4 
  bes4 c4 a4 
  g2. 
  c4 a4 f4   %35
  d'4 e4 f4 
  a,2 g8 f8 
  f2. 
  
}

alto = \relative c' {
  c4 f4 f4 
  g2. 
  f4 g4 f4 
  f2 f4 
  g8 a8 bes2   %5
  a2. 
  f2 f4 
  f8 d8 e4 f4 
  e2 fis4 
  g4 f4 e4   %10
  f8 e8 d4 e4 
  f2 g4 
  f2 f4 
  g2. 
  c2 c4   %15
  bes4 f4 bes4 
  bes2 bes4 
  a2 e4 
  a4. g8 f8 e8 
  d4. cis8 d4   %20
  d2 cis4 
  a2. 
  c4 a4 d4 
  d4 c4 bes4 
  d2 d4   %25
  d4 c4 bes4 
  g'4 e4 f4 
  g4 f4 e4 
  d4 e8 f8 g4 
  g2 e4   %30
  f2 f4 
  f2 f4 
  g2 f4 
  f4 e2 
  f2 f4   %35
  f4 g4 a4 
  f2 e4 
  c2. 
  
}

tenor = \relative c' {
  a4 c4 d4 
  e2. 
  c4 d4 c4 
  bes4 f4 bes4 
  bes8 c8 d4 c4   %5
  c2. 
  d2 d4 
  c2 c4 
  c2 c4 
  d4 c4 c4   %10
  c4 bes8 a8 bes4 
  c2 c4 
  c2 d4 
  e2. 
  g2 f4   %15
  f8 e8 d8 cis8 d4 
  g2 f4 
  e2 cis4 
  d2 c4 
  bes4 f'8 e8 d8 e8   %20
  f4 d4 a4 
  f2. 
  a2 a4 
  bes4 c4 d4 
  a4 g4 fis4   %25
  g4 fis4 g4 
  g2 a4 
  c4 b4 c4 
  c2 b4 
  c2 g4   %30
  c2 c4 
  bes2 c4 
  d4 c4 c4 
  c2. 
  c2 c4   %35
  bes4 bes4 c4 
  c4 d8 c8 bes8 c8 
  a2. 
  
}

baixo = \relative c, {
  f4 f'4 d4 
  c4 c'4 bes4 
  a4 g4 a4 
  bes4 a4 bes4 
  g2 c,4   %5
  f2 f,4 
  d'4 c4 bes4 
  a4 g4 f4 
  c'4 bes4 a4 
  g4 a4 bes4   %10
  a4 bes4 g4 
  f4 f'4 e4 
  f4 e4 d4 
  c2. 
  c'4 bes4 a4   %15
  bes4 bes,4 a'4 
  g4 f4 g4 
  a4 a,4 g'4 
  f4 e4 d4 
  g4 a4 bes4   %20
  a4 g4 a4 
  d,2. 
  a4 d4 c4 
  bes4 a4 g4 
  fis'4 e4 d4   %25
  g4 d4 g,4 
  e'4 c4 f4 
  e4 d4 c4 
  d4 g4 g,4 
  c2 c4   %30
  a2 f4 
  bes4 bes'4 a4 
  g4 e4 f4 
  c4 c'4 bes4 
  a4 f4 a4   %35
  bes4 g4 f4~ 
  f8 a,8 bes4 c4 
  f,2. 
  
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