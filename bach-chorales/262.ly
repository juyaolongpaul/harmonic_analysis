\version "2.10.33"

\header {
  title = "262 - Ach Gott, vom Himmel sieh' darein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    bes4 a4 g4 d'4 
    d8 c8 bes4 a4 c4 
    bes4 a4 g4 c4 
    bes4 a4 g4 } g4 
  a4 f4 e4 d8 e8   %5
  f4 g4 a4 f4 
  g8 a8 bes4 a4 bes4 
  g4 fis4 g4 g4 
  d'4 bes4 c4 d4 
  c4 bes4 a2   %10
  a2. 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 fis4 
    g4 fis4 g4 bes4 
    a4 g4 fis4 aes4 
    g4 fis4 cis4 d4 
    d2 d4 } d4 
  e4 d4 cis4 d4   %5
  d4 d4 e4 d8 c8 
  bes8 c8 d4 c4 d4 
  d8 ees8 d4 d4 d4 
  d4 d4 ees4 f4 
  ees4 d4 cis8 d4 cis8   %10
  d2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 c4 d4 g4 
    d4 d4 d4 ees4 
    e4 d8 c8 bes4 a4 
    g4 fis16 a16 c8 bes4 } bes4
  a4 a4 a4 a4   %5
  a4 d4 cis4 a4 
  g4 f8 g8 a8 g8 fis4 
  g8 c8 a4 bes4 bes4 
  a4 g4 g4 g4 
  g2~ g8 f8 e4   %10
  fis2. 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d4 
    g4 a4 bes4 a8 g8 
    fis4 g4 d4 c4 
    cis4 d4 e4 fis4 
    g4 d4 g,4 } g'4 
  cis,4 d4 a'8 g8 f8 e8   %5
  d8 c8 bes4 a4 d4 
  ees4 d8 e8 f8 ees8 d8 c8 
  bes8 c8 d4 g,4 g'4 
  fis4 g8 f8 ees8 d8 c8 b8 
  c4 g4 a2   %10
  d2. 
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