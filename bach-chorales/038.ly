\version "2.10.33"

\header {
  title = "38 - Straf' mich nicht in deinem Zorn"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 aes4 bes4 bes4 
    ees,4 f4 g2 
    c4 d4 ees4 g,4 
    f2 ees2 }
  bes'4 f4 g2   %5
  c4. bes8 a2 
  bes4 f4 g8 a8 bes4 
  bes4 a4 bes2 
  c4 d4 ees4 g,4 
  f2 ees2   %10
  
}

alto = \relative c' {
  \repeat volta 2 {
    ees4 ees4 f8 ees8 d4 
    ees4 ees8 d8 ees2 
    ees4 f4 ees8 d8 ees4 
    ees4 d4 bes2 }
  f'8 ees8 d8 f8 ees2   %5
  c4 c4 c2 
  f4 f4. ees8 d4 
  c2 d2 
  f4 f4 ees4 ees4 
  ees4 d4 bes2   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    bes4 c4 f,4 g4 
    g4 c8 bes8 bes2 
    aes4 aes4 bes4 bes4 
    c4 bes8 aes8 g2 }
  bes4 bes4 bes2   %5
  g8 f8 e8 g8 f2 
  bes8 c8 d4 d8 c8 bes8 a8 
  g4 f4 f2 
  aes4 bes4 bes4 bes4 
  c4 bes8 aes g2   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    ees8 d8 c8 ees8 d8 c8 bes8 d8 
    c8 bes8 aes8 bes8 ees,2 
    aes8 g8 f8 aes8 g8 f8 ees8 g8 
    aes8 f8 bes4 ees,2 }
  d'8 c8 bes8 d8 ees2   %5
  e8 d8 c8 e8 f2 
  d8 c8 bes8 d8 ees8 f8 g8 f8 
  ees8 c8 f4 bes,2 
  f8 g8 aes8 f8 g8 aes8 bes8 g8 
  aes8 f8 bes4 ees,2   %10
  
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