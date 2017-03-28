\version "2.10.33"

\header {
  title = "108 - Valet will ich dir geben"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 ees4 
    bes'4 bes4 c4 d4 
    ees2 ees4 g4 
    f4 ees4 ees4 d4 
    ees2. } ees8 f8 
  g4 g4 f4 ees4   %5
  d8 c8 d4 bes4 d4 
  ees4 d4 c4 c4 
  bes2. bes4 
  g8 aes8 bes4 c4 bes4 
  bes4 aes4 g4 bes4   %10
  aes4 g4 f4 f4 
  ees1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    ees4 g4 aes4 g4 
    g2 g4 c4 
    c4 aes4 g4 f4 
    g2. } g8 aes8 
  bes4 c4 c4 a4   %5
  bes8 a8 bes4 f4 bes4 
  bes4 bes4 bes4 a4 
  f2. f4 
  ees8 f8 g4 aes4 g4 
  g4 f4 ees4 ees4   %10
  ees4 ees4 ees4 d4 
  bes1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    bes4 ees4 ees4 d4 
    c2 c4 ees4 
    c4 c4 bes4 bes4 
    bes2. } bes4 
  ees4 ees4 f4 f4   %5
  f8 ees8 f4 d4 f4 
  ees4 f4 g4 f8 ees8 
  d2. bes4 
  bes4 ees4 ees4 des4 
  c2 c4 g4   %10
  aes4 bes4 c4 bes8 aes8 
  g1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 ees4 
    g4 ees4 aes4 b,4 
    c2 c4 c'4 
    aes4 f4 bes4 bes,4 
    ees2. } ees4 
  ees'8 d8 c8 bes8 a4 f4   %5
  bes2 bes,4 aes'4 
  g4 f4 ees4 f4 
  bes,2. d4 
  ees4 des4 c8 d8 e4 
  f2 c4 des4   %10
  c4 bes4 a4 bes4 
  ees,1 
  
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