\version "2.10.33"

\header {
  title = "100 - Durch Adams Fall ist ganz verderbt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 g4 f4 g4 
    ees4 d4 c4 g'8 a8 
    bes4 c4 g4 a4 
    bes4 a4 g4 } bes4 
  bes4 bes8 aes8 g8 f8 ees8 f8   %5
  g4 f4 ees4 g4 
  g4 g4 f4 ees8 d8 
  d2 c4 g'4 
  g4 f4 g4 a4 
  bes8 a8 g4 f4 c'4   %10
  d4 c4 g8 a8 bes4 
  a4 g4 a2 
  g2. 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    ees4 ees4 ees4 d4 
    d8 c8 c8 b8 c4 ees4 
    d4 c8 d8 ees4 ees4 
    d4 c4 b4 } g'4 
  f8 ees8 d8 c8 bes4 c8 d8   %5
  ees4 d4 ees4 bes8 c8 
  d4 c4 c4 c4 
  c4 b4 c4 d4 
  ees4 ees8 d8 ees8 g8 f8 ees8 
  d8 c8 bes4 a4 f'4   %10
  f4 ees8 d8 c4 d4 
  d8 c8 bes4 ees4 d8 c8 
  b2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    c4 c4 c4 g4 
    g4 g8 f8 ees4 c'4 
    bes8 a8 g4 c4 c4 
    fis,8 g4 fis8 g4 } d'4 
  bes4 f4 g4 g8 aes8   %5
  bes4. aes8 g4 g4 
  g8 f8 ees4 f8 g8 aes4 
  g4. f8 ees4 g4 
  c8 bes8 aes4 bes4 c4 
  f,4. e8 f4 a4   %10
  bes8 a8 g8 fis8 g4 g4 
  fis4 g2 fis4 
  g2. 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 b4 
    c8 d8 ees4 a,4 b4 
    c4 g4 c4 c4 
    g'8 f8 ees8 d8 c8 bes8 a8 g8 
    d'2 g,4 } g'4 
  d8 c8 bes4 ees8 d8 c4   %5
  g8 aes8 bes4 ees,4 ees'4 
  b4 c8 bes8 aes8 g8 f4 
  g2 c4 b4 
  c4 f4 ees8 d8 c4 
  bes4 c4 f,4 f'4   %10
  bes,4 c8 d8 ees4 bes8 c8 
  d4 ees8 d8 c4 d4 
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