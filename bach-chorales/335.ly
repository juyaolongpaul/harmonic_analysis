\version "2.10.33"

\header {
  title = "335 - Es ist das Heil uns kommen her"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    c4 c4 c8 d8 ees4 
    d4 c4 bes4 c8 bes8 
    a8 g8 f8 g8 a4 b4 
    c8. d16 d4 c4 } c4 
  f4 e4 d4 e4   %5
  f8 e8 d4 c4 c4 
  f4 c4 d4 a8 bes8 
  c4 bes4 a4 a4 
  g8 a8 bes4 a4 g4 
  d4 e4 f4   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 f4 
    f4 e4 f4 g4 
    f4 f8 ees8 d4 c4 
    c4 c4 f8 a8 g8 f8 
    e8. f16 g8. f16 e4 } f8 g8 
  a4 g4 g8 f8 e8 g8   %5
  c,8 a'8 d,8 g8 e4 f8 g8 
  a8 bes8 a8 g8 f4 f4 
  e4 d8 e8 fis4 f4 
  e4 d4 c8 d8 e4 
  d4 c8 bes8 a4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  bes8 
    c8 d8 c8 bes8 a4 g8 a8 
    bes4 a4 bes4 g4 
    f8 g8 a8 bes8 c4 d4 
    g,8 c4 b8 c4 } a4 
  a8 b8 c4 b8 a8 g4   %5
  f8 c'8 b4 c4 a8 bes8 
  c4 f8 e8 d4 d4 
  g,8 a8 bes8 c8 d4 c4 
  c4 f,8 g8 a8 bes8 c4 
  bes8 a8 g4 f4   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 f4 
    a,8 bes8 c4 f,4 c'4 
    d8 ees8 f4 g4 e4 
    f4 f,4 f'8 e8 d4 
    c4 g4 c4 } f8 e8 
  d4 e8 f8 g4 c,8 bes8   %5
  a8 f8 g4 c4 f4 
  f,8 g8 a4 bes8 c8 d4 
  e8 fis8 g4 d4 a8 bes8 
  c4 d8 e8 f4 e8 f8 
  g4 c,4 f,4   %10
  
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