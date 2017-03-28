\version "2.10.33"

\header {
  title = "109 - Da Christus geboren war"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key g \minor
}


soprano = \relative c'' {
  g4 g8 a8 bes4 
  a2 g4 
  fis8 g8 g4. fis16 g16 
  a2. 
  a8 bes8 bes4. a16 bes16   %5
  c2 d4 
  bes4 a2 
  g2. 
  a4 a4 bes4 
  c2 bes4   %10
  a4 g2 
  f2. 
  c'4 d4 e4 
  f4 e2 
  d2 cis4   %15
  d2. 
  d2 d4 
  c8 bes8 a8 bes8 c4 
  d4 c4 bes4 
  a2.   %20
  d2 d4 
  c4 d4 ees4 
  d4 c4 bes4 
  a2. 
  a8 bes8 bes4. a16 bes16   %25
  c2 d4 
  bes4 a2 
  g2. 
  
}

alto = \relative c' {
  d4 g8 fis8 g4 
  g4 fis4 g4 
  c,4 d4 cis4 
  d2. 
  fis4 g2   %5
  g4 a4 fis4 
  d4 ees4 d4 
  d2. 
  f4 f4 f4 
  e4 f4 g4   %10
  f2 e4 
  c2. 
  a'4 g2 
  f4 bes4 a4 
  f4 bes4 a4   %15
  a2. 
  bes2 bes4 
  a8 g8 f8 g8 a4 
  a4 fis4 g4 
  fis2.   %20
  g4 a4 bes4 
  a4 g4 fis4 
  g4 fis4 g4 
  fis2. 
  fis8 g8 g4. fis16 g16   %25
  a8 g8 fis8 g8 a4 
  g2 fis4 
  d2. 
  
}

tenor = \relative c' {
  bes8 c8 d4 d4 
  ees4 d8 c8 bes4 
  a4 g8 a8 bes4 
  fis2. 
  d'2 d4   %5
  c8 bes8 a8 g8 a4 
  g2 fis4 
  bes2. 
  d4 d4 d4 
  c8 bes8 a4 g4   %10
  a4 d4 c4 
  a2. 
  a4 b4 cis4 
  d2 cis4 
  d4 e2   %15
  fis2. 
  f2 f4 
  f2 ees4 
  d2 d4 
  d2.   %20
  d4 c4 bes4 
  c2 c4 
  bes4 a4 g4 
  d'2. 
  d2 d4   %25
  c4 ees4 d4 
  d4 ees4 d8 c8 
  b2. 
  
}

baixo = \relative c' {
  g8 a8 bes8 a8 g4 
  c,4 d4 ees4 
  a,4 bes4 g4 
  d'2. 
  d4 g4 f4   %5
  e4 fis4 d4 
  g4 c,4 d4 
  g,2. 
  d'8 e8 f8 e8 d4 
  a'4 d,4 e4   %10
  f4 bes,4 c4 
  f,2. 
  f'2 e4 
  d4 g8 f8 g8 a8 
  bes4 g4 a4   %15
  d,2. 
  bes8 c8 d8 c8 bes4 
  f'2 f4 
  fis4 d4 g4 
  d2.   %20
  bes'4 fis4 g4 
  a2 a,4 
  bes2 c4 
  d2. 
  d'4 c4 bes4   %25
  a4 g4 fis4 
  g4 c,4 d4 
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