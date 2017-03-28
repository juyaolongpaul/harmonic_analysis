\version "2.10.33"

\header {
  title = "74 - O Haupt voll Blut un Wunden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    d4 c4 bes4 a4 
    g2 a4 e'4 
    f4 f4 e8 d8 e4 
    d2. } f4 
  e8 d8 c4 d4 e4   %5
  f2 f4 c4 
  d4 c4 bes8 a8 bes4 
  a2. f'4 
  e8 f8 g4 f4 e4 
  d2 e4 a,4   %10
  bes4 a4 g4 c4 
  a2.
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    f8 g8 a4 d,8 e8 f4 
    f4 e4 f4 a4 
    a4 a4 a4 a8 g8 
    f2. } a4 
  g4 g8 f8 f4 bes4   %5
  bes4 a8 g8 a4 a4 
  a8 g8 a4 g8 fis8 g4 
  fis2. g4 
  g4 c4 c4 c4 
  c4 b4 c4 f,8 e   %10
  d e f4 f e
  f2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d8 e8 f4 g4 c,4 
    d4 c4 c4 e4 
    e4 d4 d4 cis4 
    a2. } d8 c8 
  bes4 c4 c4 bes8 c8   %5
  d4 c8 bes8 c4 f8 ees8 
  d4 d4 d4 d4 
  d2. d4 
  c8 d8 e4 f4 g4 
  a4 g4 g4 c,4   %10
  bes4 c4 c8 bes16 a16 g8 c8 
  c2.
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  c8 
    bes4 a4 g4 f4 
    bes,4 c4 f4 cis4 
    d8 e8 f8 g8 a4 a,4 
    d2. } d4 
  g4 a4 bes8 a8 g4   %5
  f2 f4 f4 
  bes4 fis4 g4 g,4 
  d'2. b4 
  c4 c'8 b8 a4 g4 
  f4 g4 c,4 f4   %10
  g4 a8 bes8 c4 c,4 
  f2.
  
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