\version "2.10.33"

\header {
  title = "22 - Schmücke dich, o liebe Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 f4 ees4 f4 
    g8 aes8 bes4 aes2 
    g2 bes4 g4 
    aes4 g8 f8 ees8 f8 g4 
    f2 ees2 }  %5
  bes'4 c8 d8 ees4 ees4 
  d8 c8 bes8 c8 c2 
  bes2 bes4 c8 d8 
  ees4 ees4 d8 c8 bes8 c8 
  c2 bes2   %10
  f4 g4 aes4 aes4 
  g8 bes8 aes8 g8 f4 f4 
  bes4 g4 aes4 g8 f8 
  ees8 f8 g4 f2 
  ees1   %15
  
}

alto = \relative c' {
  \repeat volta 2 {
    ees4 d4 c4 bes4 
    bes4 ees2 d4 
    ees2 f4 ees4 
    ees4 ees8 d8 c8 d8 ees4~ 
    ees4 d4 bes2 }   %5
  g'4 f4 ees8 f8 g4 
  g4 d4 g4 f8 ees8 
  d2 ees4 ees8 f8 
  g4 g4 f4 f8 g8 
  f8 ees16 d16 ees4 d2   %10
  d4 ees4 d8 ees8 f4 
  ees8 g8 f8 ees8 d4 d4 
  d4 ees4. d8 ees8 d8 
  c8 d8 ees2 d4 
  bes1   %15
  
}

tenor = \relative c' {
  \repeat volta 2 {
    bes4 bes8 aes8 g4 f4 
    ees8 f8 g4 f8 ees8 f4 
    ees2 bes'4 bes4 
    c4 bes4 c4 bes4 
    bes8 aes16 g16 aes4 g2 }  %5
  ees'4 a,8 b8 c4 c4 
  bes8 a8 bes2 a4 
  bes2 g4 aes4 
  bes8 a8 bes8 c8 d8 ees8 f8 bes,8 
  bes4 a4 bes2   %10
  bes4 bes4 aes8 g8 f4 
  bes4 c4 d8 c8 bes4 
  bes4 bes4 aes4 bes4 
  c4 bes4 bes8 aes16 g16 aes4 
  g1   %15
  
}

baixo = \relative c {
  \repeat volta 2 {
    ees4 bes4 c4 d4 
    ees4 g,8 aes8 bes2 
    ees,2 d'4 ees4 
    c8 d8 ees4 aes,4 g8 aes8 
    bes2 ees,2 }  %5
  ees'4 d4 c8 d8 ees8 f8 
  g4. f8 ees4 f4 
  bes,2 ees4 aes,4 
  g8 f8 g8 a8 bes8 c8 d8 ees8 
  f4 f,4 bes2   %10
  bes'4 aes8 g8 f4 ees8 d8 
  ees4 aes,4 bes4 bes4 
  g4 c4 f,4 g4 
  aes4 g8 aes8 bes2 
  ees,1   %15
  
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