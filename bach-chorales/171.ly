\version "2.10.33"

\header {
  title = "171 - Schaut, ihr Sünder"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  g4 a4 bes4 g4 
  r4 d'4 c4 bes4 
  a4 g4 a2 
  bes4 c4 d4 bes4 
  r4 d4 ees4 d4   %5
  c4 c4 bes2 
  d4 c4 bes4 g4 
  r4 c4 c4 bes4 
  a4 g4 f2 
  bes4 c4 d4 bes4   %10
  r4 d4 c4 g8 a8 
  bes4 a4 g2 
  
}

alto = \relative c' {
  d4 d4 d4 d4 
  r4 g4. fis8 g4~ 
  g8 fis8 g8 g,8 d'2 
  f4 f4 f4 f4 
  r4 bes4. a8 bes4   %5
  bes4 a4 f2 
  f4. fis8 g4 d4 
  r4 g4. f16 e16 f4~ 
  f4 e4 c2 
  g'4 a4 bes8 aes8 g4   %10
  r4 g4 g4 g4~ 
  g4 fis4 d2 
  
}

tenor = \relative c' {
  bes4 a4 g4 bes4 
  r4 bes4 c4 d4 
  d4. c8 fis,2 
  d'4 c4 bes4 d4 
  r4 f4 ees4 f4   %5
  g4 f8 ees8 d2 
  bes4 c4 d4 bes4 
  r4 c8 bes8 a4. g8 
  a8 bes8 c8 bes8 a2 
  d4 f4 f4 ees4   %10
  r4 f4 ees4 bes8 c8 
  d4. c8 b2 
  
}

baixo = \relative c' {
  g4. fis8 g4 g,4 
  r4 g4 a4 bes8 c8 
  d4 ees4 d2 
  d8 bes4 a8 bes4 bes'4 
  r4 bes8 aes8 g4 f4   %5
  ees4 f4 bes,2 
  bes8 bes'4 a8 g4 g,4 
  r4 e'4 f4 d4 
  c4 c4 f,2 
  g'4 f8 ees8 d8 bes8 ees4   %10
  r4 b4 c8 d8 ees4 
  d4 d,4 g2 
  
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