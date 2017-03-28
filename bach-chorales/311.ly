\version "2.10.33"

\header {
  title = "311 - Dank sei Gott in der Höhe"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    a4 c4 d4 c4 
    bes2 a4 a4 
    g4 a4 g4 g4 
    f2. } g4 
  g4 a4 bes4 c8 bes8   %5
  a2 g4 g4 
  a8 b8 c4 d8 c8 b4 
  c2. c4 
  a4 c4 d4 c4 
  bes2 a4 a4   %10
  g4 a4 g4 g4 
  f2. 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    f8 g8 a4 bes8 a8 g8 a8 
    bes8 a8 g4 f4 f4 
    e4 f4 f4 e4 
    c2. } c8 d8 
  e4 c4 d4 c4   %5
  c4. d8 e4 e4 
  f4 g4 a4 g4 
  g2. e4 
  c4 f4 f4 ees4 
  d4. e8 fis4 d4   %10
  e4 f4 f4 e4 
  c2. 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    c4 ees4 f4 e4 
    f4 c4 c4 c4 
    c4 c4 c8 bes16 a16 bes4 
    a2. } g4 
  c4 f,4 bes8 a8 g4   %5
  a8 g8 f4 c'4 c4 
  c8 d8 e4 d4. e16 f16 
  e2. g,4 
  a4 a4 bes4 a4 
  bes8 c8 d4 d4 f4   %10
  c4 c4 c8 bes16 a16 bes4 
  a2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    f8 ees8 d8 c8 bes4 c4 
    d4 e4 f4 f4 
    bes8 a8 g8 f8 c'4 c,4 
    f2. } e8 d8 
  c8 c'8 bes8 a8 g8 f8 e4   %5
  f4 a,8 bes8 c4 c4 
  f8 e8 d8 c8 f4 g4 
  c,2. c4 
  f8 ees4 d16 c16 bes8 a'8 g8 fis8 
  g8 a8 bes8 c8 d4 d8 c8   %10
  bes8 a8 g8 f8 c'4 c,4 
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