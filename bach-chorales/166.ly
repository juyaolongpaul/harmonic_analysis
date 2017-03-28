\version "2.10.33"

\header {
  title = "166 - Es stehn vor Gottes Throne"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    bes4 g4 fis4 g4 
    a2 bes4 bes4 
    a4 bes4 c4 a4 
    bes2 a4 bes4 
    a4 g4 g4 fis4     %5
    g2. } bes4 
  bes4 bes4 a4 bes4 
  g2 fis4 bes4 
  bes4 bes4 a4 bes4 
  g2 fis4 g4   %10
  a8 b8 c4 c4 b4 
  c4 a4 bes4 a4 
  g4 fis4 g
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    g4 d4 d8 c8 bes4 
    ees4 d4 d4 g4~ 
    g8 fis8 g4. e8 d8 fis8 
    d4 e4 fis4 g4 
    g8 fis8 e4 d2     %5
    d2. } g4 
  g4 g4. f8 f4~ 
  f8 ees16 d16 c4 d4 d4 
  e8 f8 g4 g4 f4~ 
  f8 ees16 d16 ees4 d4 g4   %10
  f4 g8 aes8 g4 g4 
  g4 g4 fis8 g8 a4 
  d,8 ees8 d4 d
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    d4 bes4 a4 g4 
    g4. fis8 g4 d'4 
    d4 d4 e8 c8 d8 c8 
    bes8 a8 g4 d'4 d4 
    ees8 a,8 bes4. a16 g16 a4     %5
    bes2. } d4 
  d4 d4 d4 d4~ 
  d8 c16 bes16 a8 g8 a4 g8 a8 
  bes8 d8 c4 c4 d4 
  bes4 a4 a4 c4   %10
  c8 d8 ees4 d8 c8 d4 
  ees4 e4 d4. c8 
  bes4 a8 c8 b4 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g4 
    g8 a8 bes8 c8 d4 ees8 d8 
    c4 d4 g,4 g'4 
    d4 g8 f8 e8 a8 fis8 d8 
    g4 cis,4 d4 g,4 
    c4 cis4 d4 d,4     %5
    g2. } g4 
  g8 a8 bes8 c8 d4 bes4 
  ees2 d4 g4~ 
  g8 f8 e8 c8 f8 ees8 d8 bes8 
  ees8 d8 c8 a8 d4 e4   %10
  f4 ees8 f8 g4 g,4 
  c4 cis4 d8 e8 fis4 
  g8 c,8 d4 g,4 
  
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