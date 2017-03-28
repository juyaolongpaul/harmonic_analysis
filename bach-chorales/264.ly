\version "2.10.33"

\header {
  title = "264 - Jesu, meines Herzens Freud'"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  d4 d8 ees8 f8 ees8 d4 
  c4. c8 c2 
  d4 ees8 d8 c4. bes8 
  bes2 a8 g8 a8 bes8 
  c8 bes8 a4 g4. g8   %5
  g2 a4 bes8 a8 
  g4. f8 f2 
  d'4 d8 ees8 f4 ees8 d8 
  c4. c8 c2 
  d4 ees8 d8 c4. bes8   %10
  bes2 ees8 d8 ees8 f8 
  d8 c8 d8 ees8 c4. bes8 
  bes1 
  
}

alto = \relative c' {
  f4 g4 f4 bes8 a8 
  g4 f8 e8 f2 
  f4 ees8 f8 g4 f16 ees16 d16 ees16 
  d2 c4 f4 
  f4 f4 f8 e8 d4   %5
  e2 e4 d4 
  d4 c4 c2 
  c4 bes4 c4 f4 
  g8 f8 g4 f2 
  f4 ees8 f8 g4 a4   %10
  g2 g4 f4 
  f4 bes4 bes4 a4 
  f1 
  
}

tenor = \relative c' {
  bes4 bes8 c8 d8 c8 bes4 
  bes4 a8 g8 a2 
  bes4 bes4 bes4 a4 
  f2 f4 c'4 
  c4 c4 c4 c8 b8   %5
  c2 a8 g8 f4 
  f4 e4 a2 
  a'4 g4 f4 bes,4 
  ees4. ees8 ees2 
  d8 c8 bes4 bes4 a8 d8   %10
  d2 c4 c4 
  c4 bes4 g'4 f8 ees8 
  d1 
  
}

baixo = \relative c' {
  bes8 a8 g4 d4 g8 f8 
  e8 d8 c4 f2 
  bes8 a8 g8 f8 ees4 f4 
  bes,2 f'8 e8 f8 g8 
  a8 g8 a8 bes8 c4 g4   %5
  c,2 cis4 d8 c8 
  bes4 c4 f2 
  fis4 g4 a4 bes4 
  bes8 a8 bes8 g8 a2 
  bes8 a8 g8 f8 e4 fis4   %10
  g2 c8 bes8 a8 f8 
  bes8 a8 g8 f8 ees4 f4 
  bes,1 
  
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