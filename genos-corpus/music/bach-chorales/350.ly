\version "2.10.33"

\header {
  title = "350 - Jesu, meiner Seelen Wonne"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    d4 ees4 f4 f4 
    ees4 d4 c4 c4 
    d4 ees4 f4 d4 
    c4. c8 bes2 }
  c4 d4 ees4 ees4   %5
  d4. c8 c2 
  ees4 f4 g4 g4 
  f4 f8. ees16 ees2 
  d4 ees4 f4 bes,4 
  ees4 d4 c4 c4   %10
  d4 ees4 f4 d4 
  c4 c4 bes4 bes4 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    bes4 bes4 a4 bes8 a8 
    g8 a8 bes4 a4 a4 
    bes4 a8 g8 f4 g4 
    g4 f4 f2 }
  a4 b4 c4 g4   %5
  g4 f8 ees8 ees2 
  g4 bes4 bes4 bes4 
  bes4 bes8 aes8 g2 
  g4 g4 f4 ees4 
  f4 f4 f4 f4   %10
  f4 g4 f4 f4 
  g4 f4 f4 f4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    f4 bes,8 c8 d4 d4 
    ees4 f4 f4 f4 
    f4 ees4 d8 c8 bes4 
    bes4 a4 d2 }
  f4 f,4 g4 c4   %5
  c8 a8 b4 g2 
  c4 d4 ees4 ees4 
  ees8 c8 d4 bes2 
  bes4 bes4 bes8 aes8 g4 
  c4 bes4 a4 a4   %10
  bes4 bes4 c4 bes4 
  bes4 a4 d4 d4 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    bes4 g4 d4 g4 
    c,4 bes4 f'4 f4 
    bes,4 c4 d4 g8 f8 
    ees4 f4 bes,2 }
  f'4 d4 c8 d8 ees8 f8   %5
  g4 g,4 c2 
  c'4 bes8 aes8 g8 f8 ees4 
  bes'4 bes,4 ees2 
  g4 f8 ees8 d4 ees4 
  a,4 bes4 f'4 f4   %10
  bes4 a8 g8 a8 f8 bes4 
  ees,4 f4 bes,4 bes4 
  
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