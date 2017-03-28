\version "2.10.33"

\header {
  title = "260 - Es ist gewißlich an der Zeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 bes4 
    bes8 c8 d4 c4 bes4 
    c4 c4 d4 bes8 c8 
    d4 ees4 f8 ees8 d4 
    c2 bes4 } d4 
  d4 c4 bes4 a4   %5
  bes8 c8 d4 c4 c4 
  bes4 a4 bes4 f4 
  bes4 c4 d4 bes8 c8 
  d4 ees4 f8 ees8 d4 
  c2 bes4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 f8  ees8 
    f4 bes4 a4 g4 
    g4 f4 f4 f4 
    bes4 bes4 c4 bes4 
    bes4 a4 f4 } f4 
  f4 f8 ees8 d8 g8 g8 fis8   %5
  g8 a8 bes4 a4 a4 
  a8 g8 g8 fis8 g4 d4 
  d4 g4 fis4 d4 
  g4 g4 f4 f4 
  g4 f8 ees8 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  c8 
    bes4 f'4 f8 ees8 d4 
    c8 bes8 bes8 a8 bes4 d8 ees8 
    f4 g4 f4 f4 
    g4 f8 ees8 d4 } bes4 
  bes4 a4 bes4 c4   %5
  d8 ees8 f4 f4 d4 
  d4 d4 d4 d8 c8 
  bes8 a8 g4 a4 g8 a8 
  bes4 bes bes bes 
  bes a f   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 bes8  c8 
    d4 c8 bes8 f'4 g8 f8 
    ees4 f4 bes,4 bes4 
    bes'4 a8 g8 a4 bes4 
    ees,4 f4 bes,4 } bes8 c8 
  d8 ees8 f4 g4 a4   %5
  g4 d8 ees8 f4 fis4 
  g4 d4 g8 a8 bes8 a8 
  g8 f8 ees4 d4 g,4 
  g'4 f8 ees8 d8 c8 bes4 
  ees f bes,   %10
  
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