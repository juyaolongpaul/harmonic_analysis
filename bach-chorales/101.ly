\version "2.10.33"

\header {
  title = "101 - Herr Christ, der einge Gottes-Söhn"
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
    a2 g4 d'4 
    ees4 c4 d4 c4 
    bes2. } bes8 c8 
  d4 d4 ees4 ees4   %5
  f4 ees8 d8 c4 f4 
  d4 d4 c4 bes4 
  a2 g4 d'4 
  ees4 c4 d4 c4 
  bes1   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    g8 a8 bes4 a4 g4 
    g4 fis4 g4 g4 
    g4 f4 f4 f4 
    f2. } f4 
  bes4 bes8 a8 g4 g4   %5
  f2 f4 f4 
  f4 f4 f8 ees8 d4 
  ees8 c8 d4 d4 g4 
  g4 f4 f8 g8 a4 
  f1   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    ees4 f4 f8 ees8 d4 
    ees4 d8 c8 bes4 bes4 
    bes4 a4 bes4 bes8 a8 
    d2. } d8 ees8 
  f4 g4 g8 f8 ees8 d8   %5
  c4 bes4 a4 c4 
  bes4 bes4 a4 a8 g8 
  g4 fis4 bes4 b4 
  c8 d8 ees8 a,8 bes4 f'8 ees8 
  d1   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    ees,4 bes4 f'4 g4 
    c,4 d4 ees4 g4 
    c,4 f8 ees8 d8 ees8 f4 
    bes,2. } bes4 
  bes'8 a8 g8 f8 ees8 d8 c8 bes8   %5
  a4 bes4 f4 a4 
  bes8 c8 d8 ees8 f8 fis8 g8 bes,8 
  c8 a8 d4 g,4 g'4 
  c8 bes8 a8 f8 bes8 ees,8 f8 f,8 
  bes1   %10
  
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