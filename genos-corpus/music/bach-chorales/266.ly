\version "2.10.33"

\header {
  title = "266 - Herr Jesu Christ, du höchstes Gut"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 fis4 g4 a4 
    bes4 a4 g4 g4 
    a4 a4 d4 c4 
    bes2 a4 } d4 
  c4 bes4 a4 c4   %5
  bes4 a4 g4 bes4 
  a4 g4 f8 e8 d8 e8 
  f4 g4 a4 g8 a8 
  bes4 c4 d4. c8 
  bes4 c8 bes8 a2   %10
  g2.
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    ees4 d4 d4 c4 
    bes8 c8 d8 c8 bes4 d8 e8 
    f4 f4 bes4. a8 
    a8 g16 fis16 g4 fis4 } d8 ees8 
  f4. e8 f4 g4   %5
  g4 fis4 d4 e4 
  f4 e8 d8 cis4 d4 
  d4 c4 c4 c4 
  d4 ees4 d8 fis8 g8 a8 
  bes8 a8 g2 fis4   %10
  d2.
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    a4 a4 g4 d'4 
    d8 e8 fis4 g4 bes,4 
    c4 d8 ees8 f4 c8 d8 
    g,8 d8 d'4 d4 } f,8 g8 
  a4 bes4 c4 c4   %5
  d4 d8 c8 bes4 c4 
  c4 bes4 a8 g8 f4 
  bes4 a8 g8 f4 g4 
  f4 g8 a8 bes4. a8 
  d4 c8 d8 ees4 d8 c8   %10
  b2.
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    c,4 d4 e4 fis4 
    g4 d4 g,4 g'4 
    f8 ees8 d8 c8 bes8 d8 e8 fis8 
    g8 a8 bes8 c8 d4 } bes4 
  a4 g4 f4 ees4   %5
  d8 c8 d4 g,4 c4 
  f,4 g4 a4 bes8 c8 
  d4 e4 f4 ees4 
  d4 c4 bes8 d8 e8 fis8 
  g8 f8 ees8 d8 c4 d4   %10
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