\version "2.10.33"

\header {
  title = "182 - Wär' Gott nicht mit uns diese Zeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g8  a8 
    bes4 b4 c4 d4 
    c4 b4 c4 c4 
    bes8 a8 g8 a8 bes8 c8 d4 
    c8 bes8 c4 bes4 } c4 
  d4 c8 bes8 a4 bes4   %5
  c8 bes8 a4 g4 g4 
  c4 b4 c4 d8 c8 
  bes8 a8 g4 f4 f4 
  bes4 c4 d4 c8 bes8 
  a8 g8 a4 g   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    g4 g4 g4 g4 
    ees4 d4 c4 g'8 a8 
    bes8 bes,4 c8 d8 ees8 f8 g8 
    a8 bes4 a8 f4 } f8 ees8 
  d8 e8 f8 g8 c,4 f4   %5
  ees4 a,8 d8 d4 d4 
  g8 aes8 g8 f8 ees4 d8 fis8 
  g8 d4 cis8 d4 d4 
  d4 g4 fis8 d8 g4 
  g4 fis4 d   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes8  c8 
    d4 d4 ees4 d4 
    g,8 g'4 f8 ees4 ees4 
    f4 g4 f8 ees8 d8 bes8 
    f'4 g8 f16 ees16 d4 } c4~ 
  c8 bes8 a8 g8 f4. g8   %5
  a8 g4 fis8 bes4 b4 
  c4 d4 g,8 a8 bes8 a8 
  g8 a8 bes8 a8 a4 bes4 
  bes8 a8 g4 a8 bes16 c16 d4 
  d4. c8 b4  %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g'8 aes8 g8 f8 ees8 d8 c8 b8 
    c8 ees8 g8 g,8 c4 c4 
    d4 ees4 d8 c8 bes8 ees8~ 
    ees8 d8 ees8 f8 bes,4 } a4 
  bes8 c8 d8 e8 f8 ees8 d4   %5
  c4 d4 g,4 g'8 f8 
  ees4 d4 c4 g4 
  g'8 f8 e8 a,8 d4 bes'8 a8 
  g8 f8 ees8 d8 c8 bes8 a8 g8 
  d'4 d,4 g   %10
  
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