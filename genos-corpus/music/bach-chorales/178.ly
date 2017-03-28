\version "2.10.33"

\header {
  title = "178 - Das neugeborne Kindelein"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key g \minor
}


soprano = \relative c'' {
  g4 g4 g4 
  d'4. c8 bes4 
  d4 c4 bes4 
  a2. 
  d4 d4 e4   %5
  f2 e4 
  d4 cis2 
  d2. 
  d4 c4 a4 
  bes4. c8 d4   %10
  c4 bes4 a4 
  bes2. 
  f'4 g4 f4 
  d4. c8 bes4 
  a4 g4 fis4   %15
  g2. 
  
}

alto = \relative c' {
  d4 g8 a8 bes4 
  a2 g4 
  g4 fis4 g4 
  fis2. 
  g8 a8 bes4 bes4   %5
  a2 g4 
  f4 e2 
  fis2. 
  g4 ees4 d4 
  d4 g4 f4   %10
  g4 f4 f4 
  f2. 
  bes4 bes4 a4 
  a2 g4 
  ees4 d4 d4   %15
  d2. 
  
}

tenor = \relative c' {
  bes4 d4 d4 
  d2 d8 c8 
  bes4 c4 d4 
  d2. 
  d4 g4 g4   %5
  f4 b,4 cis4 
  d4 a2 
  a2. 
  bes4 a4 a4 
  g4 ees'4 d4   %10
  ees4 d4 c4 
  d2. 
  d4 c4 c4 
  a4 d4 d4 
  c4 bes4 a4   %15
  b2. 
  
}

baixo = \relative c' {
  g8 a8 bes4 a8 g8 
  fis2 g4 
  bes,4 a4 g4 
  d'2. 
  bes'4 a8 g8 cis,4   %5
  d2 e4 
  f8 g8 a4 a,4 
  d,2. 
  g4 g'4 fis4 
  g4. a8 bes4   %10
  ees,4 f4 f,4 
  bes2. 
  bes'4 ees,4 f4 
  fis2 g4 
  c,4 d4 d4   %15
  g,2. 
  
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