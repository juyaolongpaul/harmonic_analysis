\version "2.10.33"

\header {
  title = "297 - Jesu, der du meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  d4 d4 a4 bes4 
  c4 bes8 a8 a4 g4 
  bes4 bes4 a4 g4 
  fis4 g4 a2 
  d4 d4 a4 bes4   %5
  c4 bes8 a8 a4 g4 
  bes4 bes4 a4 g4 
  fis4 g4 a2 
  a4 bes4 c4 a4 
  bes4 a8 g8 g4 f4   %10
  bes4 c4 d4 d4 
  ees4 d8 c8 c4 bes4 
  d4 d4 c4 d4 
  bes4 c4 d2 
  d4 d4 c4 bes4   %15
  a4 a4 g2 
  
}

alto = \relative c'' {
  g4 g4 a4 g4 
  g4 g4 fis4 d4 
  d4 ees4 ees8 d8 d8 c8 
  c4 d4 d2 
  d4 g4 fis4 g4   %5
  g4 g4 fis4 d4 
  d4 ees4 ees8 d8 d8 c8 
  c4 d4 d2 
  fis4 g4 g4 f4 
  f4 f4 e4 c4   %10
  f4 g8 a8 bes4 bes4 
  bes4 bes4 a4 f4 
  f4 f4 f8 ees8 d4 
  d4 g4 fis2 
  g4 g4 a4 g4   %15
  g4 fis4 d2 
  
}

tenor = \relative c' {
  bes8 c8 d4 d4 d4 
  c8 d8 ees4 d8 c8 bes4 
  g4 g4 g8 fis8 g4 
  a4 bes4 fis2 
  bes8 c8 d4 d4 d4   %5
  c8 d8 ees4 d8 c8 bes4 
  g4 g4 g8 fis8 g4 
  a4 bes4 fis2 
  d'4 d4 c4 c4 
  bes8 c8 d4 c8 bes8 a4   %10
  bes4 ees4 f4 f4 
  ees8 f8 g4 f8 ees8 d4 
  bes4 bes4 a4 a4 
  g8 bes8 a8 g8 a2 
  bes8 c8 d4 ees4 d4   %15
  ees4 d8 c8 b2 
  
}

baixo = \relative c' {
  g8 a8 bes4 fis4 g4 
  ees4 c4 d4 g,4 
  g'8 f8 ees8 d8 c8 d8 ees4 
  a,4 g4 d'2 
  g,8 a8 bes8 c8 d4 g4   %5
  ees4 c4 d4 g,4 
  g'8 f8 ees8 d8 c8 d8 ees4 
  a,4 g4 d'2 
  d4 g8 f8 e4 f8 ees8 
  d8 c8 bes4 c4 f4   %10
  d4 c4 bes4 bes'8 a8 
  g4 ees4 f4 bes,4 
  bes8 c8 d8 ees8 f4 fis4 
  g8 f8 ees4 d2 
  g8 a8 bes4 fis4 g4   %15
  c,4 d4 g,2 
  
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